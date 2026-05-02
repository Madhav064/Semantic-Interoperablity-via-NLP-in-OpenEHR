import asyncio
from playwright.async_api import async_playwright
import os

async def generate_pdf():
    html_path = r"c:\Users\madha\Desktop\OpenEHR Project\report.html"
    pdf_path = r"c:\Users\madha\Desktop\OpenEHR Project\OpenEHR_NLP_Report.pdf"
    
    file_url = f"file:///{html_path.replace(os.sep, '/')}"
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        print("Navigating to HTML file...")
        await page.goto(file_url, wait_until="networkidle")
        
        print("Generating PDF...")
        await page.pdf(
            path=pdf_path,
            format="A4",
            print_background=True,
            margin={"top": "0cm", "right": "0cm", "bottom": "0cm", "left": "0cm"}
        )
        
        await browser.close()
        print(f"Successfully generated {pdf_path}")

if __name__ == "__main__":
    asyncio.run(generate_pdf())
