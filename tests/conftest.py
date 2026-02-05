"""
Pytest configuration and fixtures
"""

import pytest


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (may require external resources)"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test (isolated, no external dependencies)"
    )


@pytest.fixture(scope="session")
def sample_resume_text():
    """Fixture providing sample resume text for testing"""
    return """
    John Doe
    Senior Machine Learning Engineer
    john.doe@email.com | +1-555-0123 | linkedin.com/in/johndoe
    
    PROFESSIONAL SUMMARY
    Experienced ML Engineer with 6 years in building production-scale machine learning systems.
    Specialized in NLP, computer vision, and deep learning architectures.
    
    TECHNICAL SKILLS
    Programming: Python, R, Java, C++, SQL
    ML/AI: TensorFlow, PyTorch, Scikit-learn, Keras, XGBoost
    NLP: spaCy, NLTK, Transformers, BERT, GPT
    Cloud: AWS (SageMaker, Lambda, EC2), GCP, Azure ML
    Tools: Docker, Kubernetes, Git, MLflow, Apache Spark
    
    PROFESSIONAL EXPERIENCE
    
    Senior ML Engineer - Tech Innovations Inc. (2021 - Present)
    - Led development of NLP-based customer sentiment analysis system processing 1M+ reviews daily
    - Implemented deep learning models achieving 94% accuracy on product recommendation engine
    - Optimized model inference reducing latency by 60% through quantization and pruning
    - Mentored team of 4 junior ML engineers
    
    Machine Learning Engineer - DataCorp Solutions (2018 - 2021)
    - Built end-to-end ML pipelines for fraud detection in financial transactions
    - Developed computer vision models for automated document verification
    - Deployed models to AWS SageMaker with CI/CD automation
    
    Data Scientist - StartupX (2017 - 2018)
    - Analyzed user behavior data to drive product recommendations
    - Created predictive models for customer churn with 85% accuracy
    
    PROJECTS
    
    Open Source Contribution - scikit-learn
    Contributed feature engineering utilities and documentation improvements
    
    Personal Project: Real-time Object Detection
    Built YOLOv5-based object detection system deployed on edge devices
    
    EDUCATION
    
    Master of Science in Computer Science
    Stanford University, 2017
    Focus: Machine Learning and Artificial Intelligence
    
    Bachelor of Engineering in Computer Science
    University of California, Berkeley, 2015
    
    CERTIFICATIONS
    
    - AWS Certified Machine Learning - Specialty
    - TensorFlow Developer Certificate
    - Deep Learning Specialization (Coursera)
    """
