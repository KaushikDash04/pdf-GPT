css = '''
<style>
.header{
position: fixed
top: 3rem;
}
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 45px;
  max-height: 45px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  font-size: 5em 
  padding: 0 1.5rem;
  color: #fff;
}
''' 

ai_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://lh3.googleusercontent.com/a/ACg8ocIUxqzS3jUdV5pU2fpLnoRVSWcgs4n7afBGuMHzG5lWxGk=s360-c-no">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://lh3.googleusercontent.com/a/ACg8ocI_-jAeg32ziiLMVHFVp1DWVrryQoX783h4nYhbkxG1wQ=s360-c-no">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''