# Library Management System using Python and SQLite3
# Mini Project

import sqlite3
from datetime import datetime

# ---------------- Database Setup ----------------

def create_connection():
    return sqlite3.connect("library.db")


def create_tables():
    conn = create_connection()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS books (
        book_id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        author TEXT NOT NULL,
        quantity INTEGER NOT NULL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS members (
        member_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        phone TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS issued_books (
        issue_id INTEGER PRIMARY KEY AUTOINCREMENT,
        book_id INTEGER,
        member_id INTEGER,
        issue_date TEXT,
        return_date TEXT,
        FOREIGN KEY(book_id) REFERENCES books(book_id),
        FOREIGN KEY(member_id) REFERENCES members(member_id)
    )
    """)

    conn.commit()
    conn.close()

# ---------------- Book Functions ----------------

def add_book(title, author, quantity):
    conn = create_connection()
    cur = conn.cursor()
    cur.execute("INSERT INTO books (title, author, quantity) VALUES (?, ?, ?)",
                (title, author, quantity))
    conn.commit()
    conn.close()
    print("Book added successfully")


def view_books():
    conn = create_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM books")
    rows = cur.fetchall()
    conn.close()

    print("\nAvailable Books:")
    for row in rows:
        print(row)

# ---------------- Member Functions ----------------

def add_member(name, phone):
    conn = create_connection()
    cur = conn.cursor()
    cur.execute("INSERT INTO members (name, phone) VALUES (?, ?)", (name, phone))
    conn.commit()
    conn.close()
    print("Member added successfully")


def view_members():
    conn = create_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM members")
    rows = cur.fetchall()
    conn.close()

    print("\nLibrary Members:")
    for row in rows:
        print(row)

# ---------------- Issue & Return ----------------

def issue_book(book_id, member_id):
    conn = create_connection()
    cur = conn.cursor()

    cur.execute("SELECT quantity FROM books WHERE book_id=?", (book_id,))
    result = cur.fetchone()

    if result and result[0] > 0:
        cur.execute("INSERT INTO issued_books (book_id, member_id, issue_date) VALUES (?, ?, ?)",
                    (book_id, member_id, datetime.now().strftime("%Y-%m-%d")))
        cur.execute("UPDATE books SET quantity = quantity - 1 WHERE book_id=?", (book_id,))
        conn.commit()
        print("Book issued successfully")
    else:
        print("Book not available")

    conn.close()


def return_book(issue_id):
    conn = create_connection()
    cur = conn.cursor()

    cur.execute("SELECT book_id FROM issued_books WHERE issue_id=? AND return_date IS NULL", (issue_id,))
    result = cur.fetchone()

    if result:
        book_id = result[0]
        cur.execute("UPDATE issued_books SET return_date=? WHERE issue_id=?",
                    (datetime.now().strftime("%Y-%m-%d"), issue_id))
        cur.execute("UPDATE books SET quantity = quantity + 1 WHERE book_id=?", (book_id,))
        conn.commit()
        print("Book returned successfully")
    else:
        print("Invalid issue ID")

    conn.close()

# ---------------- Menu ----------------

def menu():
    while True:
        print("""
------ Library Management System ------
1. Add Book
2. View Books
3. Add Member
4. View Members
5. Issue Book
6. Return Book
7. Exit
""")

        choice = input("Enter choice: ")

        if choice == '1':
            title = input("Book Title: ")
            author = input("Author: ")
            quantity = int(input("Quantity: "))
            add_book(title, author, quantity)

        elif choice == '2':
            view_books()

        elif choice == '3':
            name = input("Member Name: ")
            phone = input("Phone: ")
            add_member(name, phone)

        elif choice == '4':
            view_members()

        elif choice == '5':
            book_id = int(input("Book ID: "))
            member_id = int(input("Member ID: "))
            issue_book(book_id, member_id)

        elif choice == '6':
            issue_id = int(input("Issue ID: "))
            return_book(issue_id)

        elif choice == '7':
            print("Thank you!")
            break
        else:
            print("Invalid choice")


# ---------------- Main ----------------

if __name__ == "__main__":
    create_tables()
    menu()
