from vector_rag.config import Config
from vector_rag.db import DBFileHandler
from vector_rag.embeddings import OpenAIEmbedder
from vector_rag.model import File


def main():

    config = Config()
    print(config.DB_URL)
    # Initialize the handler with a mock embedder for testing
    handler = DBFileHandler(config, embedder=OpenAIEmbedder(config))

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
        print(f"Added file: {file.name}")
        print(f"File size: {file.file_size} bytes")
    else:
        print("Failed to add file")

    # for project in handler.get_projects():
    #     print(project.name)


if __name__ == "__main__":
    main()