import sys, io, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from pypdf import PdfReader

pdf = r"C:\Users\neo29\.claude\projects\c--Users-neo29-workspace-SVC-NSVB-ZH\ce03e666-73c3-4457-b425-1aa542c41163\tool-results\webfetch-1776335467000-ccrsxx.pdf"
reader = PdfReader(pdf)
print(f"Total pages: {len(reader.pages)}")

# Search every page for loss-related content
for i, page in enumerate(reader.pages):
    text = page.extract_text() or ""
    if re.search(r'\b(loss|L_|KL|ELBO|reconstruction|mapping|adversarial|posterior|prior)\b',
                 text, re.IGNORECASE):
        print(f"\n{'='*70}\n=== Page {i+1} ===\n{'='*70}")
        print(text[:3000])
