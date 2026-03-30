from config import dense_index, SMTP_PORT, SENDER_EMAIL, PASSWORD, SMTP_SERVER
import smtplib
from email.message import EmailMessage
import ssl

def ragQueryTool(query):
    """
    Perform a RAG query on the dense index.
    """
    
    try:
        results = dense_index.search(
            namespace="chatpdf",
            query={
                "top_k": 10,
                "inputs": {
                    "text": query
                }
            },
            rerank={
                "model": "bge-reranker-v2-m3",
                "top_n": 10,
                "rank_fields": ["chunk_text"]
            }   
        )
        
        return results
    except Exception as e:
        print(f"Error performing RAG query: {e}")
        return None
    
    
def sendEmailTool(recipient_email: str, subject: str, body: str):
    """
    A function for sending emails.
    """
    if(not recipient_email or not subject or not body):
        return "Recipient email, subject, and body are required to send an email."
    
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = SENDER_EMAIL
    msg['To'] = recipient_email
    msg.set_content(body)
    
    try:
        context = ssl.create_default_context()

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls(context=context)
            server.login(SENDER_EMAIL, PASSWORD)
            server.send_message(msg)   
            print("Email sent successfully!")
        
        return "Email sent successfully!"
    except Exception as e:
        print(f"Error sending email: {e}")
        return "Failed to send email."

        