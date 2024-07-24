from chatbot import dapatkan_respon

# Berinteraksi dengan chatbot
while True:
    input_user = input("Anda: ")
    if input_user.lower() == "keluar":
        break
    response = dapatkan_respon(input_user)
    print(f"Chatbot: {response}")
