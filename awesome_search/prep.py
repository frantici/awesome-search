"""
AwesomeSearch Prep - Tool to convert resources into markdown for awesomeSearch indexing.
"""

import argparse
import sys
import os

def cmd_prep():
    """Entry point for awesomePrep command."""
    parser = argparse.ArgumentParser(
        description='Prepare documents (PDF, DOCX, Web, etc.) into chunking-aligned Markdown for AwesomeSearch.'
    )

    # Input/Output
    parser.add_argument('input', help='Path to the input file or URL')
    parser.add_argument('output', help='Path to the output Markdown file')

    # LLM Settings
    parser.add_argument('--model', default='gpt-4o', help='LLM model name (default: gpt-4o)')
    parser.add_argument('--api-key', default=os.getenv('OPENAI_API_KEY', ''), help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--base-url', help='Custom base URL for local LLMs or alternative APIs')

    # Vision Support
    parser.add_argument('--vision', action='store_true', help='Use vision capabilities (converts pages to images for the LLM)')

    args = parser.parse_args()

    print(f"Running AwesomeSearch Prep...")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Model: {args.model}")
    print(f"Vision enabled: {args.vision}")

    try:
        chunks = get_document_chunks(args.input, args.vision)
        print(f"Extracted {len(chunks)} chunks/pages from the input.")

        client = init_llm_client(args.api_key, args.base_url)

        # Ensure output directory exists
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        with open(args.output, 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(chunks):
                print(f"Processing chunk {i+1}/{len(chunks)}...")
                is_first = (i == 0)
                formatted_markdown = process_chunk_with_llm(client, args.model, chunk, is_first)

                # Write to file and flush immediately
                f.write(formatted_markdown + "\n\n")
                f.flush()

        print(f"Preparation complete! Output saved to {args.output}")

    except Exception as e:
        print(f"Error during preparation: {e}", file=sys.stderr)
        sys.exit(1)

import fitz  # PyMuPDF
import docx
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
import base64
from typing import List, Dict, Union
import mimetypes

def encode_image(image_bytes: bytes) -> str:
    """Encode image bytes to base64 string."""
    return base64.b64encode(image_bytes).decode('utf-8')

def extract_pdf_pages(file_path: str, use_vision: bool) -> List[Union[str, Dict]]:
    """
    Extracts content from a PDF.
    If use_vision is True, yields base64 encoded images of pages.
    Otherwise, yields extracted text per page.
    """
    pages_content = []
    doc = fitz.open(file_path)

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)

        if use_vision:
            pix = page.get_pixmap()
            img_bytes = pix.tobytes("png")
            base64_img = encode_image(img_bytes)
            pages_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}})
        else:
            text = page.get_text()
            if text.strip():
                pages_content.append(text)

    return pages_content

def extract_docx_text(file_path: str) -> List[str]:
    """Extracts text from a DOCX file, chunking by roughly a page or sections."""
    doc = docx.Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        if para.text.strip():
            full_text.append(para.text)

    # Naive chunking for DOCX: combine into ~2000 character chunks
    chunks = []
    current_chunk = ""
    for p in full_text:
        current_chunk += p + "\n"
        if len(current_chunk) > 2000:
            chunks.append(current_chunk)
            current_chunk = ""
    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def extract_web_content(url: str) -> List[str]:
    """Extracts text from a URL and converts to basic markdown."""
    response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    response.raise_for_status()

    soup = BeautifulSoup(response.content, 'html.parser')

    # Remove script and style elements
    for script in soup(["script", "style", "nav", "footer", "header"]):
        script.extract()

    html_content = str(soup)
    markdown_text = md(html_content, heading_style="ATX")

    # Chunk the markdown
    lines = markdown_text.split('\n')
    chunks = []
    current_chunk = ""
    for line in lines:
        current_chunk += line + "\n"
        if len(current_chunk) > 3000 and line.strip() == "":
            chunks.append(current_chunk)
            current_chunk = ""
    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def extract_text_file(file_path: str) -> List[str]:
    """Extracts text from a plain text or markdown file, chunking it."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Simple chunking by double newlines or length
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    for p in paragraphs:
        current_chunk += p + "\n\n"
        if len(current_chunk) > 3000:
            chunks.append(current_chunk)
            current_chunk = ""
    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def get_document_chunks(input_source: str, use_vision: bool) -> List[Union[str, Dict]]:
    """
    Determines input type and extracts its content as a list of chunks/pages.
    """
    if input_source.startswith('http://') or input_source.startswith('https://'):
        return extract_web_content(input_source)

    if not os.path.exists(input_source):
        raise FileNotFoundError(f"Input file not found: {input_source}")

    mime_type, _ = mimetypes.guess_type(input_source)
    ext = os.path.splitext(input_source)[1].lower()

    if ext == '.pdf' or mime_type == 'application/pdf':
        return extract_pdf_pages(input_source, use_vision)
    elif ext == '.docx' or mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        return extract_docx_text(input_source)
    else:
        # Fallback to text reading for .txt, .md, .mdx etc
        return extract_text_file(input_source)

from openai import OpenAI

SYSTEM_PROMPT = """
You are an expert programming resource curator and document preparer for 'AwesomeSearch', a local semantic search engine.

Your task is to take the provided document content (text or image representation) and output it in a clean, standardized Markdown format perfectly optimized for semantic chunking.

Follow these strict guidelines:

1. CLASSIFICATION & METADATA (Output this at the very beginning of the document once, if it is the first chunk):
   - Resource Type: [Book / Article / Other]
   - Programming Languages: [List languages present, or "None"]

2. CONTENT FILTERING:
   - For Books: Ignore and DO NOT output boilerplate, setup instructions, table of contents, index, or copyright pages. Focus only on the core reference material, explanations, and code examples.
   - For Articles/Other: Keep the core content, ignoring navigational or irrelevant web clutter.

3. MARKDOWN FORMATTING (Optimized for AwesomeSearch):
   - Use standard Markdown headings (`#`, `##`, `###`) to denote sections clearly.
   - Separate paragraphs with a single blank line.
   - Ensure all code examples are enclosed in proper markdown code blocks (```language ... ```) specifying the correct programming language.
   - Do NOT wrap the entire output in a single code block. Just format the content as pure markdown.

4. CONTINUATION:
   - You may be receiving a part (page/chunk) of a larger document. Just process what is given to you and format it correctly according to the rules above. Do not add introductory or concluding remarks (e.g., "Here is the converted text...").
"""

def init_llm_client(api_key: str, base_url: str = None) -> OpenAI:
    """Initialize the OpenAI client."""
    client_args = {"api_key": api_key}
    if base_url:
        client_args["base_url"] = base_url
    return OpenAI(**client_args)

def process_chunk_with_llm(client: OpenAI, model: str, content: Union[str, Dict], is_first_chunk: bool) -> str:
    """Sends the chunk to the LLM to format as Markdown."""

    # Construct the user message based on content type
    if isinstance(content, dict) and content.get("type") == "image_url":
        user_message = [
            {"type": "text", "text": "Please convert this page to standard markdown following the system instructions."},
            content
        ]
    else:
        user_message = content

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + ("\nThis is the FIRST chunk of the document. Please include the Classification & Metadata block." if is_first_chunk else "\nThis is a continuation chunk. DO NOT output the Classification & Metadata block, just format the content.")},
        {"role": "user", "content": user_message}
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
    )

    return response.choices[0].message.content.strip()


if __name__ == '__main__':
    cmd_prep()
