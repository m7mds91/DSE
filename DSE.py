import os
from flask import Flask, request, render_template
from markupsafe import Markup
import fitz  # PyMuPDF
import docx
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Step 1: Load the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 2: Specify the Folder Path Containing the Documents
folder_path = "enter your path folder"  # Update this path to the folder containing your files & don't forget to add "\\" not"\"

# Step 3: Function to extract text from files
def extract_text_from_file(file_path):
    ext = file_path.lower().split('.')[-1]
    try:
        if ext == 'txt':
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read()
        elif ext == 'pdf':
            with fitz.open(file_path) as doc:
                text = ""
                for page in doc:
                    text += page.get_text()
                return text
        elif ext in ['doc', 'docx']:
            # Skip temporary files that start with ~$
            if os.path.basename(file_path).startswith('~$'):
                return ""
            doc = docx.Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return ""

# Step 4: Read All Files from the Folder and Subfolders
documents = []
file_contents = []  # List to store the contents of each file
file_names = []

# Use os.walk() to traverse the directory and all subdirectories
for root, dirs, files in os.walk(folder_path):
    for file_name in files:
        if file_name.lower().endswith(('.txt', '.pdf', '.doc', '.docx')):  # Check for supported formats
            file_path = os.path.join(root, file_name)
            content = extract_text_from_file(file_path)
            if content:  # Only add files that were successfully read
                documents.append(content)
                file_contents.append(content)  # Store the content
                file_names.append(file_name)  # Keep track of file names for reference

# Check if documents are loaded
if not documents:
    print("No documents found in the folder.")
    exit()
else:
    print(f"Loaded {len(documents)} documents from the folder and subfolders.")

# Step 5: Generate Embeddings for the Documents
print("Generating embeddings for the documents...")
embeddings = model.encode(documents)

# Convert embeddings to a numpy array
embedding_dim = embeddings.shape[1]  # Get the dimensionality of the embeddings
embeddings_np = np.array(embeddings, dtype='float32')

# Step 6: Create and Save the FAISS Index
index = faiss.IndexFlatL2(embedding_dim)
index.add(embeddings_np)
faiss.write_index(index, "vector_index.faiss")
print("Embeddings added to the FAISS index and saved as 'vector_index.faiss'.")

# Load the FAISS index (optional, but keeps it consistent with future reloads)
index = faiss.read_index("vector_index.faiss")

@app.route('/', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        query_text = request.form['query']
        
        # Generate embedding for the query
        query_embedding = model.encode([query_text])
        query_vector = np.array(query_embedding, dtype='float32')
        
        # Search the FAISS index
        k = 1  # Number of results to return
        distances, indices = index.search(query_vector, k)
        
        # Highlight occurrences of the query text within the document content
        highlighted_results = []
        for j, i in enumerate(indices[0]):
            content = file_contents[i]
            highlighted_content = content.replace(query_text, f'<mark>{query_text}</mark>')
            highlighted_results.append((Markup(highlighted_content), distances[0][j]))
        
        return render_template('results.html', query=query_text, results=highlighted_results)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
