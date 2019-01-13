import csv
import os

class summarizer(object):
	def __init__(self, path, headers, steps, restore = True, verbose = True):
		dirname, filename = os.path.split(path)
		if not os.path.exists(dirname):
			os.makedirs(dirname)
		exist = os.path.exists(path)
		self.f = open(path, 'a+' if restore else 'w+', newline = '')
		self.writer = csv.DictWriter(self.f, headers)
		self.headers = headers
		self.step = 0
		self.steps = steps
		if (not restore) or (not exist):
			self.writer.writeheader()
		self.buffer = []
		self.verbose = verbose
		self.cnt = dict.fromkeys(self.headers, 0)

	def __del__(self):
		self.flush()
		self.f.close()
	
	def flush(self):
		if len(self.buffer) == 0 or (len(self.buffer) == 1 and self.step == 0):
			return
		self._endline()
		if self.step == 0:
			for row in self.buffer[:-1]:
				self.writer.writerow(row)
		else:
			for row in self.buffer:
				self.writer.writerow(row)
		self.f.flush()
		self.buffer = []

	def _newline(self):
		self.buffer.append(dict.fromkeys(self.headers, 0))
		self.cnt = dict.fromkeys(self.headers, 0)
		self.step = 0

	def _endline(self):
		if len(self.buffer) == 0:
			return
		for key in self.buffer[-1]:
			if self.cnt[key] > 0:
				self.buffer[-1][key] /= self.cnt[key]

	def __call__(self, **kwargs):
		self.log(**kwargs)

	def log(self, **kwargs):
		if len(self.buffer) == 0:
			self._newline()

		for key, val in kwargs.items():
			self.buffer[-1][key] += val
			self.cnt[key] += 1
		self.step += 1

		if self.step == self.steps:
			self._endline()
			if self.verbose:
				print(self.buffer[-1])
			self._newline()
			if len(self.buffer) > 2:
				self.flush()
		
