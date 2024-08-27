# document-search-engine
This project is a document search engine that allows users to search through .txt, .pdf, .doc, and .docx files within a specified directory and its subdirectories. The search engine uses semantic embeddings generated by a pre-trained Sentence Transformer model to perform similarity searches, providing relevant document content as search results.
Key Features:
Multi-format Support: The search engine supports .txt, .pdf, .doc, and .docx files, making it versatile for different document types.
Recursive Directory Search: The tool traverses all subdirectories within the specified folder, indexing all supported files for comprehensive search coverage.
Semantic Search: Utilizes the all-MiniLM-L6-v2 model from Sentence Transformers to generate embeddings for documents, allowing for semantic searches that go beyond simple keyword matching.
Highlighted Results: The search results display the content of the matching documents with the query terms highlighted, making it easier for users to identify relevant information quickly.
FAISS Indexing: The project uses FAISS (Facebook AI Similarity Search) to efficiently index and search through large numbers of documents.
How It Works:
File Parsing: The application reads and extracts text from .txt, .pdf, .doc, and .docx files in the specified directory and subdirectories.
Text Embedding: It generates embeddings for the extracted text using a pre-trained Sentence Transformer model.
Indexing: The embeddings are indexed using FAISS for efficient similarity search.
Search Functionality: Users can input a query, and the application will return the most relevant documents based on the semantic similarity of their content to the query.
Web Interface: The application is built with Flask, providing a simple web interface for users to perform searches and view results with highlighted query terms.
Technologies Used:
Python: The core language used for the application.
Flask: A lightweight web framework used to create the search engine’s web interface.
FAISS: A library for efficient similarity search, used for indexing and searching document embeddings.
Sentence Transformers: Pre-trained models from the sentence-transformers library are used for generating semantic embeddings.
PyMuPDF (fitz): Used for extracting text from PDF files.
python-docx: Used for extracting text from DOC and DOCX files.
