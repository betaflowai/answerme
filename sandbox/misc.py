import PyPDF2


def convert_pdf_to_text(pdf_file_obj):
    pdfreader = PyPDF2.PdfReader(pdffileobj)
    n = len(pdfreader.pages)
    all_text = ""
    for i in range(n):
        page = pdfreader.pages[i]
        all_text += f"## Page {i + 1} ##\n\n===\n\n"
        all_text += page.extract_text()
        all_text += "\n\n====\n\n"
    return all_text


if __name__ == '__main__':
    pdffileobj = open('/home/mbaddar/Downloads/vfcon058470.pdf', 'rb')
    text = convert_pdf_to_text(pdffileobj)
    print(text)