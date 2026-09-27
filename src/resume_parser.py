import pdfplumber

def extract_text(pdf_file):

    text = ""

    with pdfplumber.open(pdf_file) as pdf:

        for page in pdf.pages:

            page_text = page.extract_text()

            if page_text:

                text += page_text + " "

    return text


def extract_skills(text):

    skills_database = [

        "Python",
        "Java",
        "C",
        "C++",
        "SQL",
        "HTML",
        "CSS",
        "JavaScript",
        "Machine Learning",
        "Deep Learning",
        "Artificial Intelligence",
        "Data Science",
        "Pandas",
        "NumPy",
        "Scikit-Learn",
        "TensorFlow",
        "PyTorch",
        "Flask",
        "Django",
        "React",
        "Node.js",
        "Git",
        "MongoDB",
        "MySQL",
        "Power BI",
        "Excel"

    ]

    found_skills = []

    text = text.lower()

    for skill in skills_database:

        if skill.lower() in text:

            found_skills.append(skill)

    return found_skills