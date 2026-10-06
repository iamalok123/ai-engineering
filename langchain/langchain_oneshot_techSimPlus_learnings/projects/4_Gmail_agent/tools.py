### Sending Email Agent - send_email

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from langchain.tools import tool
from dotenv import load_dotenv
load_dotenv()
import os



# 1. Define configuration variables
smtp_server = "smtp.gmail.com"
smtp_port = 587              
sender_email = os.getenv("SENDER_EMAIL")
password = os.getenv("PASSWORD")  


def send_email_by_gmail(to:str, subject:str, body_text:str):
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = to
    message["Subject"] = subject

    message.attach(MIMEText(body_text, "plain"))

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
    
        server.login(sender_email, password)
        
        server.sendmail(sender_email, to, message.as_string())
        print("Email sent successfully!")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # 6. Clean up and close connection safely
        server.quit()


# # Test the function
# send_email_by_gmail("69alokhotta@gmail.com", "Test email", "Test email body from LangChain!")




@tool
def send_email(to:str, subject:str, body:str):
    """
    Send Email to any email address by providing the correct email address, subject and body.
    Args:
        to - email address
        subject - email subject in one line
        body - email body in plane text
    """
    send_email_by_gmail(to, subject, body_text=body)
    return "Email Send Successfully..."

ALL_TOOLS = [send_email]