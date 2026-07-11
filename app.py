import streamlit as st
import PyPDF2
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .score-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .skill-tag {
        display: inline-block;
        background: #4CAF50;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

# Skill Database
SKILLS_DB = {
    'Programming Languages': ['python', 'java', 'javascript', 'c++', 'c', 'ruby', 'php', 'swift', 'go', 'rust', 'scala', 'kotlin', 'r', 'matlab', 'sql'],
    'Web Development': ['html', 'css', 'react', 'angular', 'vue', 'django', 'flask', 'node.js', 'express', 'spring', 'laravel', 'rails'],
    'Data Science': ['machine learning', 'deep learning', 'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas', 'numpy', 'matplotlib', 'seaborn', 'data analysis', 'data visualization'],
    'Cloud & DevOps': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git', 'ci/cd', 'linux', 'terraform', 'ansible'],
    'Databases': ['mysql', 'postgresql', 'mongodb', 'oracle', 'sqlite', 'redis', 'cassandra', 'elasticsearch'],
    'Soft Skills': ['communication', 'leadership', 'teamwork', 'problem solving', 'project management', 'agile', 'scrum', 'time management'],
    'Mobile Development': ['android', 'ios', 'react native', 'flutter', 'swift', 'kotlin', 'xamarin'],
    'AI/ML': ['natural language processing', 'nlp', 'computer vision', 'neural networks', 'reinforcement learning', 'transformers', 'bert']
}

JOB_DESCRIPTIONS = {
    'Data Scientist': 'Required skills: Python, Machine Learning, Deep Learning, TensorFlow, PyTorch, Pandas, NumPy, SQL, Data Analysis, Statistics, Data Visualization, Scikit-learn',
    'Web Developer': 'Required skills: HTML, CSS, JavaScript, React, Node.js, Express, MongoDB, Git, REST APIs, Responsive Design, Web Development',
    'DevOps Engineer': 'Required skills: Docker, Kubernetes, AWS, Azure, CI/CD, Jenkins, Linux, Terraform, Ansible, Git, Python, Bash scripting',
    'Software Engineer': 'Required skills: Java, Python, C++, Data Structures, Algorithms, OOP, Git, SQL, Problem Solving, Software Design',
    'AI Engineer': 'Required skills: Python, Machine Learning, Deep Learning, NLP, Computer Vision, TensorFlow, PyTorch, Neural Networks, Mathematics, Statistics'
}

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    try:
        text = ""
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def extract_skills(resume_text):
    """Extract skills from resume using keyword matching"""
    resume_text_lower = resume_text.lower()
    found_skills = {category: [] for category in SKILLS_DB.keys()}
    all_found_skills = []
    
    for category, skills in SKILLS_DB.items():
        for skill in skills:
            if skill.lower() in resume_text_lower:
                found_skills[category].append(skill.title())
                all_found_skills.append(skill.title())
    
    # Extract education using regex patterns
    education_patterns = [
        r'([A-Z][a-z]+ (?:University|College|Institute|School))',
        r'((?:University|College|Institute) of [A-Z][a-z]+)',
        r'(B\.(?:Tech|Sc|E|A)\.?)',
        r'(M\.(?:Tech|Sc|E|A|B\.A)\.?)',
        r'(B\.?Tech in [A-Za-z\s]+)',
        r'(M\.?Tech in [A-Za-z\s]+)',
    ]
    
    education = []
    for pattern in education_patterns:
        matches = re.findall(pattern, resume_text)
        education.extend(matches)
    
    # Remove duplicates
    education = list(set(education))
    
    # Extract years of experience
    experience_patterns = [
        r'(\d+)\+?\s*(?:years?|yrs?)',
        r'(\d+)\+?\s*years?\s*of\s*experience',
        r'experience[:\s]+(\d+)\+?\s*years?',
    ]
    
    max_experience = 0
    for pattern in experience_patterns:
        matches = re.findall(pattern, resume_text_lower)
        if matches:
            years = [int(m) for m in matches]
            max_experience = max(max_experience, max(years))
    
    return found_skills, all_found_skills, education, max_experience

def calculate_resume_score(resume_text, job_description=None):
    """Calculate resume score based on various factors"""
    score = 0
    scoring_details = {}
    
    found_skills, all_skills, education, experience = extract_skills(resume_text)
    total_skills_found = len(all_skills)
    
    # Skills score (0-40)
    skill_score = min(total_skills_found * 3, 40)
    score += skill_score
    scoring_details['Skills'] = f"{skill_score}/40"
    
    # Education score (0-20)
    education_score = min(len(education) * 10, 20)
    score += education_score
    scoring_details['Education'] = f"{education_score}/20"
    
    # Experience score (0-20)
    experience_score = min(experience * 4, 20)
    score += experience_score
    scoring_details['Experience'] = f"{experience_score}/20"
    
    # Content quality score (0-20)
    word_count = len(resume_text.split())
    if word_count >= 300:
        content_score = 20
    elif word_count >= 200:
        content_score = 15
    elif word_count >= 100:
        content_score = 10
    else:
        content_score = 5
    score += content_score
    scoring_details['Content Quality'] = f"{content_score}/20"
    
    # Job match score
    job_match_score = 0
    if job_description:
        job_match_score = calculate_job_match(resume_text, job_description)
        scoring_details['Job Match'] = f"{job_match_score}/100"
    
    return {
        'total_score': min(score, 100),
        'details': scoring_details,
        'skills': all_skills,
        'education': education,
        'experience': experience,
        'job_match': job_match_score if job_description else None
    }

def calculate_job_match(resume_text, job_description):
    """Calculate job match using TF-IDF and cosine similarity"""
    try:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        documents = [resume_text.lower(), job_description.lower()]
        tfidf_matrix = vectorizer.fit_transform(documents)
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return round(similarity * 100, 2)
    except:
        return 0

def find_best_job_match(resume_text):
    """Find best matching job roles"""
    job_scores = {}
    for job_title, job_desc in JOB_DESCRIPTIONS.items():
        match_score = calculate_job_match(resume_text, job_desc)
        job_scores[job_title] = match_score
    return sorted(job_scores.items(), key=lambda x: x[1], reverse=True)

def generate_improvements(resume_text, score_details, skills_found):
    """Generate improvement suggestions"""
    suggestions = []
    
    word_count = len(resume_text.split())
    if word_count < 300:
        suggestions.append("📝 Add more content to your resume. Aim for at least 300 words.")
    
    skill_categories_found = [cat for cat, skills in SKILLS_DB.items() 
                             if any(s.lower() in resume_text.lower() for s in skills)]
    if len(skill_categories_found) < 3:
        suggestions.append(" Add more technical skills from different categories (Programming, Cloud, Databases, etc.)")
    
    if not score_details.get('Education', '').startswith('20'):
        suggestions.append("🎓 Clearly mention your educational background and institutions.")
    
    if not score_details.get('Experience', '').startswith('20'):
        suggestions.append("💼 Add work experience details with duration and responsibilities.")
    
    action_verbs = ['developed', 'created', 'managed', 'led', 'implemented', 'designed', 'built', 'optimized']
    has_action_verbs = any(verb in resume_text.lower() for verb in action_verbs)
    if not has_action_verbs:
        suggestions.append("✍️ Use action verbs to describe your achievements (e.g., 'Developed', 'Led', 'Implemented').")
    
    if not re.search(r'\d+%', resume_text) and not re.search(r'\$\d+', resume_text):
        suggestions.append(" Add quantifiable achievements (e.g., 'Improved performance by 30%', 'Managed $50K budget').")
    
    if 'python' not in [s.lower() for s in skills_found]:
        suggestions.append("🐍 Consider learning Python - it's highly sought after in the industry.")
    
    if 'git' not in [s.lower() for s in skills_found]:
        suggestions.append("🔧 Add version control experience (Git/GitHub) to your resume.")
    
    if not suggestions:
        suggestions.append("✅ Your resume looks good! Consider adding more specific achievements and metrics.")
    
    return suggestions

# Main App
def main():
    # Header
    st.markdown('<h1 class="main-header">📄 AI Resume Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("### Upload your resume and get instant AI-powered feedback!")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("️ Settings")
        target_job = st.selectbox(
            "Select Target Job Role (Optional)",
            ["None"] + list(JOB_DESCRIPTIONS.keys())
        )
        
        st.markdown("---")
        st.markdown("### 📊 Available Job Roles")
        for job in JOB_DESCRIPTIONS.keys():
            st.markdown(f"• {job}")
    
    # File uploader
    uploaded_file = st.file_uploader("📤 Upload your resume (PDF)", type=['pdf'])
    
    if uploaded_file is not None:
        # Display file info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"📄 File: {uploaded_file.name}")
        with col2:
            st.info(f"📦 Size: {round(uploaded_file.size/1024, 2)} KB")
        with col3:
            st.success("✅ Ready to analyze")
        
        # Analyze button
        if st.button("🚀 Analyze Resume", type="primary", use_container_width=True):
            with st.spinner("🔍 Analyzing your resume..."):
                # Extract text
                resume_text = extract_text_from_pdf(uploaded_file)
                
                if resume_text.startswith("Error"):
                    st.error(f"❌ {resume_text}")
                    st.stop()
                
                # Get job description if selected
                job_desc = JOB_DESCRIPTIONS.get(target_job) if target_job != "None" else None
                
                # Calculate score
                score_result = calculate_resume_score(resume_text, job_desc)
                
                # Extract skills
                found_skills, all_skills, education, experience = extract_skills(resume_text)
                
                # Find job matches
                job_matches = find_best_job_match(resume_text)
                
                # Generate suggestions
                suggestions = generate_improvements(resume_text, score_result['details'], all_skills)
                
                # Display Results
                st.markdown("---")
                st.header("📈 Analysis Results")
                
                # Overall Score
                score = score_result['total_score']
                
                if score >= 80:
                    score_color = "#4CAF50"
                    score_emoji = "🏆"
                elif score >= 60:
                    score_color = "#FF9800"
                    score_emoji = "👍"
                else:
                    score_color = "#f44336"
                    score_emoji = ""
                
                st.markdown(f"""
                    <div class="score-box">
                        <h2>{score_emoji} Overall Score: {score}/100</h2>
                    </div>
                """, unsafe_allow_html=True)
                
                # Detailed metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Skills Found", f"{len(all_skills)}")
                with col2:
                    st.metric("Experience", f"{experience} years")
                with col3:
                    st.metric("Education", f"{len(education)}")
                with col4:
                    st.metric("Word Count", f"{len(resume_text.split())}")
                
                # Skills by category
                st.markdown("---")
                st.subheader(" Skills Extracted")
                
                for category, skills in found_skills.items():
                    if skills:
                        st.markdown(f"**{category}:**")
                        skills_html = "".join([f'<span class="skill-tag">{skill}</span>' for skill in skills])
                        st.markdown(skills_html, unsafe_allow_html=True)
                        st.markdown("")
                
                if not any(found_skills.values()):
                    st.warning("️ No skills detected. Make sure your resume clearly lists your technical skills.")
                
                # Job matching
                st.markdown("---")
                st.subheader("🎯 Job Role Matching")
                
                for job_name, match_score in job_matches:
                    st.markdown(f"**{job_name}**")
                    st.progress(match_score / 100)
                    st.caption(f"Match: {match_score}%")
                
                # Target job match
                if target_job != "None" and score_result['job_match']:
                    st.markdown("---")
                    st.success(f"🎯 Match with **{target_job}**: {score_result['job_match']}%")
                
                # Improvements
                st.markdown("---")
                st.subheader("💡 Improvement Suggestions")
                
                for i, suggestion in enumerate(suggestions, 1):
                    st.markdown(f"{i}. {suggestion}")
                
                # Download report
                st.markdown("---")
                
                # Create report
                report = f"""
AI RESUME ANALYZER REPORT
========================

Overall Score: {score}/100

DETAILED BREAKDOWN:
"""
                for category, detail in score_result['details'].items():
                    report += f"\n{category}: {detail}"
                
                report += f"\n\nSKILLS FOUND ({len(all_skills)}):\n"
                for category, skills in found_skills.items():
                    if skills:
                        report += f"\n{category}: {', '.join(skills)}"
                
                report += f"\n\nEDUCATION: {', '.join(education) if education else 'Not detected'}"
                report += f"\nEXPERIENCE: {experience} years"
                
                report += "\n\nJOB MATCHES:\n"
                for job, match_score in job_matches:
                    report += f"\n{job}: {match_score}%"
                
                report += "\n\nIMPROVEMENT SUGGESTIONS:\n"
                for i, suggestion in enumerate(suggestions, 1):
                    report += f"\n{i}. {suggestion}"
                
                st.download_button(
                    label="📥 Download Analysis Report",
                    data=report,
                    file_name=f"resume_analysis_{uploaded_file.name.replace('.pdf', '.txt')}",
                    mime="text/plain",
                    use_container_width=True
                )
    
    else:
        # Show demo/info when no file uploaded
        st.markdown("---")
        st.info(" Upload a PDF resume to get started!")
        
        st.markdown("### ✨ Features:")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **📊 Resume Scoring**
            - Skills analysis
            - Experience evaluation
            - Content quality check
            """)
        
        with col2:
            st.markdown("""
            **🎯 Job Matching**
            - Match with 5 job roles
            - Similarity scoring
            - Best fit recommendation
            """)
        
        with col3:
            st.markdown("""
            **💡 Improvements**
            - Personalized suggestions
            - Skill gaps identification
            - Actionable feedback
            """)

if __name__ == "__main__":
    main()
