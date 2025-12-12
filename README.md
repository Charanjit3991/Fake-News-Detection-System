# Fake-News-Detection-System
Capstone Project

A role-based Fake News Detection and Analysis System built using Machine Learning (Logistic Regression and TF-IDF) to identify misinformation in news headlines, full articles, and datasets. The system supports multiple stakeholder roles and provides analytics, explainability, and reporting features.

⸻

Project Overview

The Fake News Detection System is designed to help different stakeholders verify information integrity:
	•	Moderators quickly validate headlines and full articles
	•	Data Scientists analyze large datasets and misinformation trends
	•	PR Analysts monitor brand-related misinformation and generate mitigation reports
	•	Professors compare machine learning models and inspect feature importance for evaluation

The system uses two trained Logistic Regression models:
	•	Old Model: Trained on approximately 40,000 news articles
	•	New Model: Enhanced with the LIAR dataset, achieving 98.88% accuracy

⸻

Technologies Used
	•	Python
	•	Streamlit
	•	Scikit-learn
	•	TF-IDF Vectorization
	•	Pandas and NumPy
	•	BeautifulSoup and Requests
	•	Joblib

⸻

Supported User Roles and Features

Moderator
	•	Paste and classify multiple headlines
	•	Verify full news articles
	•	View real/fake prediction with confidence score
	•	Optional Google Fact Check integration

Data Scientist
	•	Upload CSV files for batch classification
	•	Analyze fake versus real distribution
	•	View keyword frequency and article length trends
	•	Export classified datasets as CSV

PR Analyst
	•	Monitor misinformation using brand keywords
	•	Analyze datasets via text paste or CSV upload
	•	Generate full reports and fake-only mitigation reports
	•	Download actionable CSV files

Professor
	•	Compare predictions from old and new machine learning models
	•	Inspect TF-IDF feature weights
	•	Export model comparison reports
	•	Evaluate system transparency and explainability

⸻

System Architecture

Processing pipeline:

Input (Headline, Article, or Dataset)
→ TF-IDF Vectorization
→ Logistic Regression Model
→ Prediction (Real or Fake with Confidence Score)
→ Analysis, Visualization, and CSV Export


Repository Structure

├── app.py
├── fake_news_model.pkl
├── new_fake_news_model.pkl
├── tfidf_vectorizer.pkl
├── new_tfidf_vectorizer.pkl
├── requirements.txt
└── README.md


