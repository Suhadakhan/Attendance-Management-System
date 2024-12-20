import sqlite3


class Tables:

    def __init__(self):
        conn = sqlite3.connect('attendance.db')
        conn.execute('PRAGMA foreign_keys=on;')
        conn.execute('''CREATE TABLE IF NOT EXISTS COURSE
                    (course_id       varchar(30)    PRIMARY KEY,
                    course_name      varchar(30)    NOT NULL);''')

        conn.execute('''CREATE TABLE IF NOT EXISTS TEACHER
                    (faculty_id varchar(30)    PRIMARY KEY,
                    ffname      varchar(30)    NOT NULL,
                    flname      varchar(30)    NOT NULL,
                    password    varchar(30)    NOT NULL,
                    course_id   varchar(30)    NOT NULL,
                    FOREIGN KEY (course_id) REFERENCES course (course_id) ON DELETE CASCADE);''')

        conn.execute('''CREATE TABLE IF NOT EXISTS STUDENT
                    (roll_no    varchar(30)    PRIMARY KEY,
                    sfname      varchar(30)    NOT NULL,
                    slname      varchar(30)    NOT NULL,
                    password    varchar(30)    NOT NULL);''')

        conn.execute('''CREATE TABLE IF NOT EXISTS ATTENDANCE
                    (roll_no    varchar(30)    NOT NULL,
                    date        DATE           NOT NULL,
                    status      BOOLEAN        NOT NULL,
                    course_id   varchar(30)    NOT NULL,
                    faculty_id  varchar(30)    NOT NULL,
                    FOREIGN KEY (roll_no) REFERENCES STUDENT (roll_no) ON DELETE CASCADE,
                    FOREIGN KEY (course_id) REFERENCES course (course_id) ON DELETE CASCADE);''')

        conn.execute('''CREATE TABLE IF NOT EXISTS STUDIES
                    (course_id  varchar(30)    NOT NULL,
                    roll_no     varchar(30)    NOT NULL,
                    faculty_id  varchar(30)    NOT NULL,
                    FOREIGN KEY (course_id)  REFERENCES course (course_id) ON DELETE CASCADE,
                    FOREIGN KEY (roll_no)    REFERENCES STUDENT (roll_no) ON DELETE CASCADE,
                    FOREIGN KEY (faculty_id) REFERENCES TEACHER (faculty_id)ON DELETE CASCADE) ;''')

        cursor = conn.execute("SELECT COUNT(*) FROM COURSE")
        cnt = cursor.fetchall()
        if cnt[0][0] == 0:
            cursor.execute(
                "INSERT INTO COURSE VALUES ('2CE1','DSA')")
            cursor.execute(
                "INSERT INTO COURSE VALUES ('2CE2','OOP')")
            cursor.execute(
                "INSERT INTO COURSE VALUES ('2CE3','DBMS')")
            cursor.execute(
                "INSERT INTO COURSE VALUES ('2CE4','OS')")
            cursor.execute(
                "INSERT INTO COURSE VALUES ('2CE5','CP2')")
            cursor.execute(
                "INSERT INTO COURSE VALUES ('2CE6','TOC')")
            cursor.execute(
                "INSERT INTO COURSE VALUES ('2CE7','AI')")
            cursor.execute(
                "INSERT INTO COURSE VALUES ('2CE8','CC')")
            cursor.execute(
                "INSERT INTO COURSE VALUES ('2CE9','SVT')")
            cursor.execute(
                "INSERT INTO COURSE VALUES ('2CE10','ITIM')")
            cursor.execute(
                "INSERT INTO COURSE VALUES ('2CE11','ASB2')")
            cursor.execute(
                "INSERT INTO COURSE VALUES ('2CE12','UED')")

        cursor = conn.execute("SELECT COUNT(*) FROM TEACHER")
        cnt = cursor.fetchall()
        if cnt[0][0] == 0:
            cursor.execute(
                "INSERT INTO TEACHER VALUES ('1', 'Nishi', 'Patwa', '2001', '2CE1')")
            cursor.execute(
                "INSERT INTO TEACHER VALUES ('2', 'Om', 'Prakash', '2002', '2CE2')")
            cursor.execute(
                "INSERT INTO TEACHER VALUES ('3', 'Megha', 'Patel', '2003', '2CE3')")
            cursor.execute(
                "INSERT INTO TEACHER VALUES ('4', 'Paresh', 'Solanki', '2004', '2CE4')")
            cursor.execute(
                "INSERT INTO TEACHER VALUES ('5', 'Rajul', 'Suthar', '2005', '2CE5')")
            cursor.execute(
                "INSERT INTO TEACHER VALUES ('6', 'Chirag', 'Patel', '2006', '2CE6')")
            cursor.execute(
                "INSERT INTO TEACHER VALUES ('7', 'Ravi', 'Raval', '2007', '2CE7')")
            cursor.execute(
                "INSERT INTO TEACHER VALUES ('8', 'Devang', 'Pandeya', '2008', '2CE8')")
            cursor.execute(
                "INSERT INTO TEACHER VALUES ('9', 'Manan', 'Thakkar', '2009', '2CE9')")
            cursor.execute(
                "INSERT INTO TEACHER VALUES ('10', 'Bhavesha', 'Suthar', '2010', '2CE10')")



        conn.commit()
        conn.close()