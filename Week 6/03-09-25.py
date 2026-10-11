# ⊛⊛⊛⊛⊛⊛⊛⊛⊛⊛⊛⊛⊛⊛⊛⊛⊛⊛⊛⊛⊛⊛⊛⊛⊛⊛⊛⊛⊛
# ⊛⊛⊛⊛⊛⊛⊛⊛⊛⊛⊛⊛⊛ ATM ⊛⊛⊛⊛⊛⊛⊛⊛⊛⊛⊛⊛⊛
# ⊛⊛⊛⊛⊛⊛⊛⊛⊛⊛⊛⊛⊛⊛⊛⊛⊛⊛⊛⊛⊛⊛⊛⊛⊛⊛⊛⊛⊛

# deadline ---> 4 sep

# pin.txt ---> 1122 ----> 1234
# bal.txt ---> 12000 --> 10000 ----> 12000

# press 1 : withdraw
    # enter a pin = 1122
        # enter a amount = 2000
            # transaction done
            # current balance = 10000
        # low balance
    # wrong pin  

# press 2 : check balance
    # enter a pin = 1122
        # current balance = 10000
    # wrong pin  

# press 3 : change pin 
    # enter a pin = 1122
        # enter a new pin = 1234
        # confirm your pin = 1234
        # pin match nhi hori(gali)
        # pin changed
    # wrong pin  

# press 4 : deposit
    # enter a pin = 1234
        # enter a amount = 2000
        # current balance = 12000
    # wrong pin


# ATM MACHINE

# print("press 1: to withdraw")
# print("press 2: check balance")
# print("press 3: to change pin")
# print("press 4: to deposit")
# print("press 0: to exit")

# option=int(input("enter your option: "))

# while option!=0:
#     match(option):
#         case 1:
#             amount=int(input("enter withdraw amount: "))
#             pin=int(input("enter your pin: "))
#             pin_file=open("Practic program\\pin.txt","r")
#             value = pin == int(pin_file.read())
#             pin_file.close()
#             if value:
#                 balance_file=open("Practic program\\balance.txt","r")
#                 bal=int(balance_file.read())
#                 value1=amount<bal
#                 if value1:
#                     print("transaction successfull")
#                     current_balance=bal-amount
#                     balance_file.close()
#                     print(f"current balance {current_balance:f}")
#                     balance_file=open("Practic program\\balance.txt","w")
#                     balance_file.write(str(current_balance))
#                     balance_file.close()
#                 else:
#                     print("insufficient balance")
#             else:
#                 print("incorrect pin")

#         case 2:
#             print("to see balance")
#             pin=int(input("enter your pin: "))
#             pin_file=open("Practic program\\pin.txt","r")
#             value2=pin == int(pin_file.read())
#             pin_file.close()
#             if value2:
#                 balance_file=open("Practic program\\balance.txt","r")
#                 bal=balance_file.read()
#                 balance_file.close()
#                 print(f"your balance is {bal}")
#             else:
#                 print("incorrect pin")

#         case 3:
#             print("to change pin")
#             pin=int(input("enter your current pin: "))
#             pin_file=open("pin.txt","r")
#             value3= pin == int(pin_file.read())
#             pin_file.close()
#             if value3:
#                 new_pin=int(input("enter new pin: "))
#                 confirm_pin=int(input("enter confirm pin: "))
#                 value4=new_pin == confirm_pin
#                 if value4:
#                     pin_file=open("Practic program\\pin.txt","w")
#                     pin_file.write(str(confirm_pin))
#                     print("pin changed")
#                     pin_file.close()
#                 else:
#                     print("enter the same pin")
#             else:
#                 print("incorrect pin")
#         case 4:
#             print("to deposit")
#             amount=int(input("enter amount to deposit: "))
#             pin=int(input("enter your pin: "))
#             pin_file=open("Practic program\\pin.txt","r")
#             value5=pin == int(pin_file.read())
#             if value5:
#                 balance_file=open("Practic program\\balance.txt","r")
#                 new_balance=amount+int(balance_file.read())
#                 balance_file.close()
#                 balance_file=open("Practic program\\balance.txt","w")
#                 balance_file.write(str(new_balance))
#                 print(f"current balance is {new_balance:d}")
#                 balance_file.close()
#             else:
#                 print("incorrect pin")
#         case 0:
#             break
#     option=int(input("enter your option: "))






#Write a program to accept marks of 6 students and display them in a sorted manner.

# l=[]
# for i in range(6):
#     lst=int(input(f"enter the marks {i+1}st subject: "))
#     l.append(lst)
# print(sorted(l))
# l=list(map(int,input("enter the marks: ")))  


# class Employee: 
#     company = "Google" # Specific to Each Class 

# harry = Employee() # Object Instatiation 
# harry.company 
# Employee.company = "YouTube" 


#---------------------Without use of file handling---------------------------

pin=1111
bal=20000

print("press 1: to withdraw")
print("press 2: check balance")
print("press 3: to change pin")
print("press 4: to deposit")
print("press 0: to exit")

option=int(input("Press the value: "))

while True:
    match(option):
        case 1:
            new_pin=int(input("Enter the pin: "))
            if new_pin==pin:
                amount=int(input("Enter the amount: "))
                if amount<bal:
                    bal=bal-amount
                    print("Your Balance is ",bal)
                    print("Sucessfull Transcition")
                    print("Visit Again")
                else:
                    print("Insuffient amount")
            else:
                print("Wrong pin!")
        case 2:
            new_pin=int(input("Enter the pin: "))
            if new_pin==pin:
                print("your Balance is ",bal)
            else:
                print("Wrong Pin!")

