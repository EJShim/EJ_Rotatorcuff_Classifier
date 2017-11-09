import smtplib

print("start")
server = smtplib.SMTP('mail.kist.re.kr')
print("serer defined")
server.starttls()
print("start ttls")
server.login("ejshim@kist.re.kr","tladmdwns!23")
print("login")

msg = "Message"
print("send message")
server.sendmail("ejshim@kist.re.kr", "ejshim@kist.re.kr", msg)
print("send message completed")
server.quit()

# server = smtplib.SMTP('mail.kist.re.kr', 25)
# server.starttls()
# server.login("G16006@kist.re.kr", "tladmdwns!23")

# msg = 'message'
# server.sendmail("G16006@kist.re.kr", "ejshim@kist.re.kr", msg)
# server.quit()