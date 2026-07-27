import os
import fitz  # PyMuPDF

out_dir = os.path.join(os.path.dirname(__file__), "sample_documents")
os.makedirs(out_dir, exist_ok=True)

# 1. Create Deep Learning PDF
doc1 = fitz.open()
page1 = doc1.new_page()
page1.insert_text((50, 50), "Deep Learning Architectures for Visual & Natural Language Processing", fontsize=16)
page1.insert_text((50, 80), "Page 1: Introduction to Residual Networks and Attention Mechanisms.", fontsize=12)
page1.insert_text((50, 100), "Convolutional Neural Networks (CNNs) have revolutionized computer vision tasks including object detection and image segmentation. The introduction of ResNet introduced skip connections allowing networks to exceed 100 layers without vanishing gradients.", fontsize=10)

page2 = doc1.new_page()
page2.insert_text((50, 50), "Deep Learning Architectures - Page 2: Transformers & LLMs", fontsize=14)
page2.insert_text((50, 80), "Transformer architectures replace recurrence with self-attention mechanisms, enabling massive parallel processing across GPUs. Models such as BERT and GPT leverage multi-head attention to model complex natural language tasks.", fontsize=10)
doc1.save(os.path.join(out_dir, "Deep_Learning_Architectures.pdf"))
doc1.close()

# 2. Create Cloud Security PDF
doc2 = fitz.open()
p1 = doc2.new_page()
p1.insert_text((50, 50), "Enterprise Cloud Security & Infrastructure Frameworks", fontsize=16)
p1.insert_text((50, 80), "Page 1: Zero Trust Architecture and Cryptographic Controls.", fontsize=12)
p1.insert_text((50, 100), "Modern cloud security demands a Zero Trust framework where no entity inside or outside the network is trusted by default. Identity and Access Management (IAM), mutual TLS, and AES-256 encryption safeguard multi-tenant cloud storage.", fontsize=10)

p2 = doc2.new_page()
p2.insert_text((50, 50), "Enterprise Cloud Security - Page 2: Container & Kubernetes Defense", fontsize=14)
p2.insert_text((50, 80), "Kubernetes cluster security requires pod security policies, network isolation, and runtime threat detection. Automated CI/CD scanning ensures vulnerability mitigation before deployment.", fontsize=10)
doc2.save(os.path.join(out_dir, "Cloud_Security_Frameworks.pdf"))
doc2.close()

print(f"Sample PDFs successfully created in {out_dir}")
