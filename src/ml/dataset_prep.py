import os
import json
from typing import Tuple, List

CATEGORIES = [
    "Artificial Intelligence",
    "Machine Learning",
    "Computer Vision",
    "Natural Language Processing",
    "Robotics",
    "Cyber Security",
    "Cloud Computing"
]

SAMPLE_DATA = {
    "Artificial Intelligence": [
        "Artificial intelligence and deep search algorithms for automated reasoning and problem solving in expert systems.",
        "General intelligence, cognitive architectures, heuristic search, and knowledge representation in modern AI.",
        "Ethical considerations in artificial intelligence, alignment of intelligent agents, and multi-agent systems research.",
        "Autonomous decision-making frameworks utilizing artificial intelligence for enterprise automation."
    ],
    "Machine Learning": [
        "Supervised learning algorithms, decision trees, random forests, gradient boosting machines, and support vector machines.",
        "Deep neural networks, backpropagation, stochastic gradient descent, cross-validation, and hyperparameter tuning.",
        "Unsupervised clustering, principal component analysis, autoencoders, and feature selection in machine learning pipelines.",
        "Reinforcement learning, Q-learning, policy gradients, and reward optimization in complex dynamic environments."
    ],
    "Computer Vision": [
        "Convolutional neural networks for object detection, semantic segmentation, image classification, and visual recognition.",
        "Feature extraction using SIFT, ORB, optical flow, camera calibration, 3D point cloud reconstruction, and photogrammetry.",
        "Face recognition, visual tracking, image synthesis using Generative Adversarial Networks (GANs), and diffusion models.",
        "Medical image processing, MRI scan segmentation, radiological image analysis using deep vision models."
    ],
    "Natural Language Processing": [
        "Transformer models, BERT, GPT architectures, attention mechanisms, tokenization, and contextual text embeddings.",
        "Sentiment analysis, named entity recognition, part-of-speech tagging, language modeling, and machine translation.",
        "Retrieval-Augmented Generation (RAG), vector embeddings, semantic retrieval, and prompt engineering for Large Language Models.",
        "Text summarization, question answering systems, dependency parsing, and topic modeling in modern NLP."
    ],
    "Robotics": [
        "Kinematics, dynamics, motion planning, trajectory optimization, and robotic arm manipulation in automated assembly.",
        "Mobile robotics, Simultaneous Localization and Mapping (SLAM), LiDAR sensor fusion, and autonomous navigation.",
        "Human-robot interaction, haptic feedback, soft robotics, exoskeleton control, and bipedal walking locomotion.",
        "Industrial automation, ROS (Robot Operating System), actuators, PID control systems, and microcontrollers."
    ],
    "Cyber Security": [
        "Network defense, intrusion detection systems, penetration testing, vulnerability assessment, and zero-trust security architecture.",
        "Cryptography, public key infrastructure, AES encryption, RSA, hash functions, and secure authentication protocols.",
        "Malware analysis, reverse engineering, ransomware mitigation, endpoint detection and response, and threat intelligence.",
        "Cloud security controls, IAM policies, Web Application Firewalls (WAF), and secure software development lifecycle."
    ],
    "Cloud Computing": [
        "Serverless computing, microservices architecture, Docker containerization, Kubernetes orchestration, and cloud infrastructure.",
        "Amazon Web Services (AWS), Microsoft Azure, Google Cloud Platform, Infrastructure as Code (IaC) using Terraform.",
        "Distributed storage systems, cloud databases, auto-scaling groups, load balancers, and multi-region high availability.",
        "DevOps pipelines, continuous integration and deployment (CI/CD), service mesh, and edge computing paradigms."
    ]
}

def generate_dataset() -> Tuple[List[str], List[int]]:
    texts = []
    labels = []
    
    # Expand dataset with synthetic variations for robust training
    for idx, category in enumerate(CATEGORIES):
        samples = SAMPLE_DATA[category]
        for text in samples:
            texts.append(text)
            labels.append(idx)
            # Add augmented sentence variations
            texts.append(f"This document discusses {category.lower()} concepts: {text}")
            labels.append(idx)
            texts.append(f"Research paper focusing on key advances in {category}: {text}")
            labels.append(idx)

    return texts, labels
