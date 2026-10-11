from dotenv import load_dotenv
load_dotenv()


#### Import all the packages
import uuid
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field
from typing import Annotated, Literal
from send_email import send_mail


#### Deine the LLM Object
llm = ChatGroq(model="openai/gpt-oss-20b")


class EmailState(BaseModel):
    question: str = ""
    mail_reason: str = ""
    recipient_name: str = ""
    recipient_email: str = ""
    subject: str = ""
    body: str = ""
    feedback: str = ""
    feedback_count: int = 0  # Max count = 3
    response: str = ""




#### Retriever Node

# UserDetails : A Pydantic schema that defines what 3 pieces of info to look for
class UserDetails(BaseModel):
    mail_reason: str = Field(description="Reason of the email, like why user want to send this email")
    recipient_name: str = Field(description="recipient / reciever name, if available, else '' ")
    recipient_email: str = Field(description="recipient / reciever email address")


def retriever_node(state:EmailState) -> EmailState:
    llm_for_user_details = llm.with_structured_output(UserDetails)
    userDetails:UserDetails = llm_for_user_details.invoke(f"Retrieve the user details from this query: {state.question}.")
    
    if not userDetails.mail_reason or not userDetails.recipient_email:
        pass
    
    state.recipient_name = userDetails.recipient_name
    state.recipient_email = userDetails.recipient_email
    state.mail_reason = userDetails.mail_reason
    
    return state






#### Draft Node

class DraftEmail(BaseModel):
    subject: str = Field(description="Email subject within 40 Words")
    body: str = Field(description="Email Body including proper details, reason and greetings and others")


def draft_node(state:EmailState) -> EmailState:
    "Draft a Email"
    draft_email_llm = llm.with_structured_output(DraftEmail)
    
    prompt = f"""
        Write a email with these details:
        To: {state.recipient_name}
        Request: {state.mail_reason}
        
        Please write a proper mail body and subject. Without extra text. Mail body
        max size will be 200 Words.  
    """
    
    if state.feedback:
        prompt = f"""Revise this email based on the feedback below.
            Current Email: 
            Subject: {state.subject}
            Body: {state.body}

            Feedback: {state.feedback}

            Please write a proper mail body and subject. Without extra text and improvement based on the feedback. 
            Mail body max size will be 200 Words.  """
    
            
    draftEmail:DraftEmail = draft_email_llm.invoke(prompt)
    state.subject = draftEmail.subject
    state.body = draftEmail.body
    
    return state






### Review Node

def review_node(state: EmailState) -> EmailState:  
    response = interrupt({
        "message": "You want to approve this or re-write this email",
        "subject": state.subject,
        "body": state.body
    })
    
    clean_resp = str(response).strip().lower()
    if clean_resp in ["yes", "y", "approve", "approved"]:
        state.feedback = ""
    elif clean_resp in ["cancel", "no", "quit", "stop"]:
        state.feedback = "cancelled"
        state.feedback_count = 99  # Force cancel
    else:
        state.feedback = response
        state.feedback_count = state.feedback_count + 1
    
    return state



### Router Node

def router(state: EmailState) -> Literal["draft", "send", "cancel"]:
    if state.feedback_count > 2 or state.feedback == "cancelled":
        return "cancel"
    if not state.feedback or state.feedback.strip() == "":
        return "send"
    
    return "draft"


### Cancel Node

def cancel_node(state: EmailState) -> EmailState:
    if state.feedback == "cancelled":
        state.response = "Email cancelled by user."
    else:
        state.response = "Email Not Sent, Because you have reached the limit of feedbacks"
    return state



### Send Node

def send_node(state: EmailState) -> EmailState:
    "Send final email"
    res = send_mail(state.recipient_email, state.subject, state.body)
    state.response = res
    return state




graph = StateGraph(EmailState)
graph.add_node("retriever", retriever_node)
graph.add_node("draft", draft_node)
graph.add_node("send", send_node)
graph.add_node("cancel", cancel_node)
graph.add_node("review_node", review_node)


graph.add_edge(START, "retriever")
graph.add_edge("retriever", "draft")
graph.add_edge("draft", "review_node")
graph.add_conditional_edges('review_node', router)
graph.add_edge("send", END)
graph.add_edge("cancel", END)



final_graph = graph.compile(checkpointer=InMemorySaver())



if __name__ == "__main__":
    while True:
        query = input("\nUser: ").strip()
        if query.lower() in ["quit", "exit"]:
            print("Bye 👋")
            break
        if not query:
            continue
        
        # Generate unique thread_id per session to prevent state leakage
        session_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": session_id}}
        res = final_graph.invoke(
            {"question": query},
            config=config
        )

        while True:
            state = final_graph.get_state(config)
            
            if not state.next:
                break
            
            print("\n" + "=" * 60)
            recipient_display = res.get("recipient_name", "")
            if res.get("recipient_email"):
                recipient_display += f" <{res.get('recipient_email')}>"
            if recipient_display:
                print("To:      ", recipient_display)
            print("Subject: ", res.get("subject", ""))
            print("-" * 60)
            print("Body:\n" + res.get("body", ""))
            print("=" * 60 + "\n")
            

            feedback = input("Approve to send (yes) or provide feedback / 'cancel': ")
            res = final_graph.invoke(
                    Command(resume=feedback),
                    config=config
                )         
        
        print("\nAI: ", res.get("response", ""))

