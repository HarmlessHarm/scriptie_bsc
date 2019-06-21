import threading


def threadsafe_generator(f):
	def g(*a, **kw):
		return threadsafe_iter(f(*a, **kw))
	return g

class threadsafe_iter:
	def __init__(self, it):
		self.it = it
		self.lock = threading.Lock()

	def __iter__(self):
		return self

	def next(self):
		with self.lock:
			return self.it.next()

@threadsafe_generator
def count():
	i = 0
	while True:
		i += 1
		yield i


class Counter:
	def __init__(self):
		self.i = 0
		self.lock = threading.Lock()

	def __iter__(self):
		return self

	def next(self):
		with self.lock:
			self.i += 1
			return self.i

def loop(func, n):
	for i in range(n):
		func()

def run(f, repeats=1000, nthreads=10):
	threads = [threading.Thread(target=loop, args=(f, repeats)) 
							for i in range(nthreads)]		

	for t in threads:
		t.start()

	for t in threads:
		t.join()

def main():
	c1 = count()
	c2 = Counter()

	run(c1.next, repeats=100000, nthreads=2)
	print("c1", c1.next())

	run(c2.next, repeats=100000, nthreads=2)
	print("c2", c2.next())	


if __name__ == '__main__':
	main()