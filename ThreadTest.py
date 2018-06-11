import threading


def func1():
	for i in range(10):
		print('func1 is called')


def func2():
	while True:
		print('func2 is called')



t1 = threading.Thread(target=func1)
t2 = threading.Thread(target=func2, daemon = True)

t1.start()
t2.start()

t1.join()
print(threading.enumerate())
# t2.join()

print("Exit the program")