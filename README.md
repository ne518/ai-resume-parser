# AI-Powered Resume Analyzer

The **AI-Powered Resume Analyzer** is an advanced NLP-based system that matches resumes against job descriptions using sophisticated text processing techniques. This tool helps job seekers optimize their resumes for Applicant Tracking Systems (ATS) by identifying missing keywords and providing actionable improvement suggestions.

---

## ✨ Features

- **📄 Multi-format Support:** Process PDF and DOCX resume files.  
- **🔍 Advanced Text Analysis:** TF-IDF vectorization with cosine similarity scoring.  
- **📊 Visual Analytics:** Interactive similarity gauge and keyword analysis.  
- **🎯 Skill Identification:** Categorizes technical, soft, and business skills.  
- **💡 Actionable Insights:** Provides specific recommendations for resume improvement.  
- **⚡ Offline Functionality:** Works without internet connection or external API dependencies.  

---

## 🛠️ Technical Architecture

The system follows a **modular pipeline**:

1. **Text Extraction:** PDF and DOCX parsing using `PyPDF2` and `python-docx`.  
2. **Text Preprocessing:** Advanced cleaning and normalization of text.  
3. **TF-IDF Vectorization:** Feature extraction with n-grams (1-3 words).  
4. **Cosine Similarity:** Matching algorithm for resume-job description comparison.  
5. **Keyword Analysis:** Smart keyword extraction and presence checking.  
6. **Skill Categorization:** Classification into technical, soft, and business skills.  

---

## Algorithms & Techniques

- **TF-IDF (Term Frequency-Inverse Document Frequency):** For keyword importance weighting.  
- **Cosine Similarity:** For semantic matching between documents.  
- **N-gram Processing:** Captures phrases and multi-word expressions.  
- **Text Normalization:** Advanced cleaning and preprocessing pipeline.  

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher  
- pip (Python package manager)  

### Steps

1. Clone the repository:

```bash
git clone https://github.com/ne518/ai-resume-parser.git
cd ai-resume-parser
