import streamlit as st
from rag.chunking import LineChunker
from rag.db import DBFileHandler
from rag.model import File
from rag.embeddings import OpenAIEmbedder
import tempfile
import os
import hashlib
import logging
from typing import Optional, List
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class ProjectManager:
    def __init__(self):
        self.handler = self._init_handler()

    @staticmethod
    @st.cache_resource
    def _init_handler():

        CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 25))
        CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 3))
        line_chunker = LineChunker(CHUNK_SIZE, CHUNK_OVERLAP)
        # Construct database URL using environment variables
        db_url = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"

        # Debug information
        logger.debug(f"Connecting to: {os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}")
        logger.debug(f"CHUNK_SIZE: {CHUNK_SIZE} ChUNK_OVERLAP: {CHUNK_OVERLAP}")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            api_key = st.secrets.get("OPENAI_API_KEY")
            
        if not api_key:
            st.error("OpenAI API key not found. Please set OPENAI_API_KEY in your environment or Streamlit secrets.")
            st.stop()
            
        # call db initializer which does not exist yet
        # then run this return DBFileHandler(db_url=db_url, embedder=OpenAIEmbedder(api_key=api_key))
        return DBFileHandler(embedder=OpenAIEmbedder(api_key=api_key), chunker=line_chunker)

    def create_project(self, name: str, description: Optional[str] = None):
        if name:
            project = self.handler.create_project(name, description)
            st.success(f"Created project: {project.name} (ID: {project.id})")
            return project
        return None

    def add_file_to_project(self, project_id: int, uploaded_file) -> bool:
        # Check if file was already uploaded in this session
        session_key = f"uploaded_{uploaded_file.name}_{project_id}"
        if session_key in st.session_state:
            logger.debug(f"File {uploaded_file.name} was already uploaded in this session")
            return True

        logger.debug(f"Processing new file upload: {uploaded_file.name}")
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            content = uploaded_file.getvalue()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        try:
            file = File(
                name=uploaded_file.name,
                path=tmp_file_path,
                content=content.decode('utf-8'),
                crc=hashlib.md5(content).hexdigest(),
                meta_data={"type": uploaded_file.type}
            )
            
            result = self.handler.add_file(project_id, file)
            if result:
                logger.debug(f"Successfully added file: {result.name}")
                st.success(f"Added file: {result.name}")
                # Mark file as uploaded in this session
                st.session_state[session_key] = True
                return True
            else:
                logger.error("Failed to add file to database")
                st.error("Failed to add file")
                return False
        finally:
            os.unlink(tmp_file_path)

    def remove_file_from_project(self, project_id: int, file_id: int) -> bool:
        try:
            logger.debug(f"Attempting to remove file with ID: {file_id} from project: {project_id}")
            result = self.handler.delete_file(file_id)
            logger.debug(f"Delete file result: {result}")
            
            if result:
                st.success(f"Removed file with ID: {file_id}")
                st.session_state.files_changed = True
                logger.debug("Set files_changed flag to True")
                return True
            else:
                st.error(f"Failed to remove file with ID: {file_id}")
                return False
        except Exception as e:
            logger.error(f"Error removing file: {str(e)}", exc_info=True)
            st.error(f"Error removing file: {str(e)}")
            return False

    def search_project(self, project_id: int, query: str, page: int, 
                      page_size: int, similarity_threshold: float):
        results = self.handler.search_chunks_by_text(
            project_id=project_id,
            query_text=query,
            page=page,
            page_size=page_size,
            similarity_threshold=similarity_threshold
        )
        return results

    def get_projects(self):
        return self.handler.get_projects()

    def list_project_files(self, project_id: int):
        logger.debug(f"Fetching files for project: {project_id}")
        files = self.handler.list_files(project_id)
        logger.debug(f"Found {len(files)} files")
        return files

class UI:
    def __init__(self):
        self.project_manager = ProjectManager()

    def render_sidebar(self):
        with st.sidebar:
            st.header("Project Operations")
            st.subheader("Create New Project")
            name = st.text_input("Project Name")
            desc = st.text_area("Project Description")
            if st.button("Create Project"):
                self.project_manager.create_project(name, desc)

    def render_file_upload(self, project_id: int):
        st.subheader("Add File")
        
        # Clear file uploader if upload was successful
        if "last_uploaded_file" in st.session_state and st.session_state.last_uploaded_file:
            st.session_state.last_uploaded_file = None
            return

        uploaded_file = st.file_uploader("Choose a file", key=f"file_upload_{project_id}")
        if uploaded_file is not None:
            logger.debug(f"File selected: {uploaded_file.name}")
            if self.project_manager.add_file_to_project(project_id, uploaded_file):
                st.session_state.last_uploaded_file = uploaded_file
                st.rerun()

    def render_file_list(self, project_id: int):
        st.subheader("Project Files")
        logger.debug(f"Rendering file list for project: {project_id}")
        files = self.project_manager.list_project_files(project_id)
        
        if not files:
            st.info("No files in this project")
            return

        # Create a dictionary to deduplicate files by their name and content hash
        unique_files = {}
        for file in files:
            file_key = f"{file.name}_{file.crc}"  # Using name and CRC as unique identifier
            if file_key not in unique_files:
                unique_files[file_key] = file
        
        logger.debug(f"Number of unique files: {len(unique_files)}")

        # Initialize session state for files_changed if not exists
        if 'files_changed' not in st.session_state:
            st.session_state.files_changed = False
            logger.debug("Initialized files_changed session state")

        for file in unique_files.values():
            col_info, col_remove = st.columns([3, 1])
            with col_info:
                st.write(f"File: {file.name} (ID: {file.id})")
            with col_remove:
                button_key = f"remove_{file.id}_{file.crc}"
                logger.debug(f"Button key: {button_key}")
                if st.button("Remove", key=button_key):
                    logger.debug(f"Remove button clicked for file: {file.name} (ID: {file.id})")
                    if self.project_manager.remove_file_from_project(project_id, file.id):
                        logger.debug("File removal successful, triggering rerun")
                        st.session_state.files_changed = True
                        st.rerun()
                    else:
                        logger.debug("File removal failed")
        
        # Check if files have changed and rerun if needed
        if st.session_state.files_changed:
            logger.debug("Files changed flag detected, triggering rerun")
            st.session_state.files_changed = False
            st.rerun()

    def render_search_interface(self, project_id: int):
        st.header("Search Documents")
        search_query = st.text_input("Enter your search query")
        similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.7, 0.05)
        page_size = st.number_input("Results per page", min_value=1, value=10)
        page = st.number_input("Page", min_value=1, value=1)
        
        if st.button("Search") and search_query:
            results = self.project_manager.search_project(
                project_id, search_query, page, page_size, similarity_threshold
            )
            self.render_search_results(results)

    def render_search_results(self, results):
        if results.total_count == 0:
            st.info("No matching results found.")
            return

        st.write(f"Found {results.total_count} matching chunks")
        st.write(f"Page {results.page} of {results.total_pages}")
        
        # Create a dictionary to deduplicate chunks based on content and index
        unique_chunks = {}
        for chunk_result in results.results:
            chunk_key = f"{chunk_result.chunk.content}_{chunk_result.chunk.index}"
            if chunk_key not in unique_chunks or chunk_result.score > unique_chunks[chunk_key].score:
                unique_chunks[chunk_key] = chunk_result

        for chunk_result in unique_chunks.values():
            with st.expander(f"Score: {chunk_result.score:.3f}"):
                st.text(chunk_result.chunk.content)
                st.write(f"Chunk Index: {chunk_result.chunk.index}")
                st.write(f"Chunk Size: {chunk_result.chunk.size} characters")

    def render_project_selection(self):
        st.header("Projects")
        projects = self.project_manager.get_projects()
        
        if not projects:
            st.info("No projects found. Create a new project using the sidebar.")
            return None

        selected_project = st.selectbox(
            "Select Project",
            options=projects,
            format_func=lambda x: f"{x.name} (ID: {x.id})"
        )
        return selected_project

    def render_project_view(self, project):
        st.subheader(f"Project: {project.name}")
        st.write(f"Description: {project.description}")
        
        col1, col2 = st.columns(2)
        with col1:
            self.render_file_upload(project.id)
        with col2:
            self.render_file_list(project.id)
        
        self.render_search_interface(project.id)

    def render(self):
        st.title("RAG Project Manager")
        self.render_sidebar()
        
        selected_project = self.render_project_selection()
        if selected_project:
            self.render_project_view(selected_project)

def main():
    ui = UI()
    ui.render()

if __name__ == "__main__":
    main()
