class ATM:
 print("Please Insert your Card")
 print("==================================")

password=111222
pin=int(input("Enter your Password"))
blance=5000
if pin==password:
    while True:

        print("""
          1 == Blance
          2 == Withdraw Blance
          3 == Deposit Blance 
          4 == Exit """)
        print("==================================")
        try:
            option = int(input("Please Enter Your Choice"))
        except:
            print("Please Enter your Valid Option")
            print("==================================")
            continue
        if option == 1:
            print("[Your Current Balance is]==", blance)
            print("==================================")
        elif option == 2:
            withdraw_ammount = int(input("Enter your Withdraw Amount"))
            if withdraw_ammount <= blance:
                blance = blance - withdraw_ammount
                print(withdraw_ammount, "is debited from your Account")
                print("Your Current Balance is ", blance)
                print("==================================")
            else:
                print("Insufficient Balance")
                print("==================================")
        elif option == 3:
            deposit_ammount = int(input("Enter your deposit amount"))
            blance = blance + deposit_ammount
            print(deposit_ammount, "is credited in your account")
            print("Your Updated Balance is ", blance)
            print("==================================")
        elif option == 4:
            break
else:
    print("Invalid Password. Please Enter your correct password")
def main():
    T=ATM()