from .models import EnvironmentState

def simulate_verify_document(state: EnvironmentState, doc_name: str) -> bool:
    if doc_name not in state.documents:
        return False
        
    doc = state.documents[doc_name]
    if doc.status == "missing":
        return False
        
    if doc.status == "submitted" or doc.status == "rejected":
        if doc.is_valid:
            doc.status = "verified"
            return True
        else:
            doc.status = "rejected"
            return False
            
    return True # already verified
