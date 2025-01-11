from rag.db import DBFileHandler
from rag.embeddings import OpenAIEmbedder
from rag.model import File


def main():
    # Initialize the handler with a mock embedder for testing
    handler = DBFileHandler(embedder=OpenAIEmbedder())

    # Create a new project
    project = handler.get_or_create_project("My RAG Project", "Testing RAG system integration")
    print(f"Created project: {project.name} (ID: {project.id})")

    # Create a test file
    test_file = File(
        name="test.txt",
        path="/path/to/test.txt",
        crc="test123",
        content="This is a test document.\nIt contains multiple lines.\nWe'll use it to test the RAG system.",
        meta_data={"type": "txt"}
    )

    # Add file to project
    file = handler.add_file(project.id, test_file)
    if file:
        print(f"Added file: {file.filename}")
        print(f"File size: {file.file_size} bytes")
    else:
        print("Failed to add file")


if __name__ == "__main__":
    main()