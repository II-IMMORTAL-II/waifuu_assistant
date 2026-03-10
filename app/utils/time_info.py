import datetime


def get_time_information() -> str:
    now = datetime.datetime.now()
    return (
        "Current Real-time Information:\n"
        f"Day: {now.strftime('%A')}\n"        # e.g. Monday
        f"Date: {now.strftime('%d')}\n"       # e.g. 05
        f"Month: {now.strftime('%B')}\n"      # e.g. February
        f"Year: {now.strftime('%Y')}\n"       # e.g.
    )