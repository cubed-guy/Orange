import orange as testee

test_id = 0
def test_many(f, tests):
	print(f'Testing {f.__qualname__}')
	for args, expected in tests: test(f, args, expected)
	print()

def test(f, args, expected):
	result = f(*args)
	if result == expected:
		print(f'[PASS] f({", ".join(repr(arg) for arg in args)}) == {result!r}')
	else:
		print(f'[FAIL] f({", ".join(repr(arg) for arg in args)}) == {result!r}, expected {expected!r}')

test_many(testee.Patterns.alias_through_strings, [
	(('name',), (None, 'name')),
	(('as name',), (None, 'as name')),
	(('aas name',), (None, 'aas name')),
	(('a as name',), ('a', 'name')),
	(('T.haxe K V as name',), ('T.haxe K V', 'name')),
	(('133 as name T.haxe K V',), ('133', 'name T.haxe K V')),
	(('bool.true{}as true',), ('bool.true{}', 'true')),
	(('as name T.haxe K V',), (None, 'as name T.haxe K V')),
])

for i in range(0, 64, 8):
	print(1<<i, '->', testee.get_discriminator_size(1<<i))
