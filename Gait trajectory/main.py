from collections import defaultdict, deque

def schedule_training(sports_data):
    schedule = defaultdict(lambda: defaultdict(list))
    student_sports = defaultdict(list)
    for sport, students in enumerate(sports_data, 1):
        for student in students:
            student_sports[student].append(sport)

    student_queue = deque(sorted(student_sports))

    while student_queue:
        student = student_queue.popleft()
        sports = student_sports[student]

        for sport in sports:
            day = len(schedule[sport]) // 2 + 1
            daily_sessions = schedule[sport].get(day, []) 

            if len(daily_sessions) == 0:
                schedule[sport][day] = [[student]]
            elif len(daily_sessions) == 1 and student not in daily_sessions[0]:  
                daily_sessions.append([student])
            elif student not in daily_sessions[0]:
                daily_sessions[0].append(student)
            else:
                student_queue.append(student)  
                break

    return schedule


def main():
    # Input handling:
    n = int(input("Enter the number of sports: "))
    sports_data = [
        list(map(int, input(f"Enter students for sport {i + 1}: ").split(",")))
        for i in range(n)
    ]

    # Schedule generation:
    result = schedule_training(sports_data)

    # Output formatting:
    for sport, days in result.items():
        for day, sessions in days.items():
            for idx, session in enumerate(sessions):
                session_name = "FN" if idx == 0 else "AN"
                print(f"Sport {sport} Day {day} {session_name}")
                print(" ".join(map(str, session)) if session else "No students available")


if __name__ == "__main__":
    main()


