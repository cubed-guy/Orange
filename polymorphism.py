# DONE: snippets / operators
# DONE: string literals
# DONE: control structures
# DONE: pointer types
# DONE: Any
# DONE: return type checking
# DONE: pointer assignment
# DONE: user-defined types
# DONE: type pattern matching
# DONE: classmethods
# TODO: array deref syntax
# TODO: array assignment syntax
# TODO: a (better) way to cast variables
# TODO: arrays
# TODO: constants
# TODO: check non-void return

from sys import argv
from enum import Enum, auto
from enum import Flag as Enum_flag
from typing import Union, Optional

class Shared:
	debug = True
	line = '[DEBUG] ** Empty line **'
	line_no = 0

class Subscriptable:
	def __getitem__(self, key):
		return f'{self.__class__.__name__}_{id(self)&0xffff:x04}[{key}]'

if __name__ != '__main__': Shared.debug = True
elif '-d' in argv: Shared.debug = True; argv.remove('-d')
else: Shared.debug = False
Shared.debug = True

WIN64, ELF64, *_ = range(4)
PTR_SIZE = 8

if   '-win' in argv: arch = WIN64; argv.remove('-win')
elif '-elf' in argv: arch = ELF64; argv.remove('-elf')
elif Shared.debug: arch = WIN64
else:
	print('Format not specified. A "-win" or "-elf" flag is required.')
	quit(1)

crlf = int(arch == WIN64)

if len(argv) <2:
	if Shared.debug: argv.append('test.poly')
	else: print('Input file not specified'); quit(1)
name = argv[1].rpartition('.')[0]
if len(argv)<3: argv.append(name+'.asm')

Shared.infile = open(argv[1])
Shared.out = open(argv[2], 'w')
def output(*args, file = Shared.out, **kwargs):
	if None in args:
		err('[Internal Error] None passed into output()')
	print(*args, **kwargs, file = file)

def err(msg):
	print(f'File "{argv[1]}", line {Shared.line_no}')
	print('   ', Shared.line.strip())
	if Shared.debug: raise RuntimeError(msg)

	print(msg)
	quit(1)

def abort():
	err('Forced Abort')

class Patterns:
	import re
	stmt = re.compile(r'(?P<indent>\s*)(?P<stmt>(?:"(?:\\.|[^"])*?"|\'(?:\\.|[^\'])*?\'|.)*?)(//|$)')
	declaration = re.compile(r'let\s+(?P<name>\w+)\s*:\s*(?P<type>\w+)')
	split_word = re.compile(r'(\w*)\s*(.*)')
	last_word = re.compile(r'(.*?)(\w+?)$')

	@classmethod
	def through_strings(cls, c):
		# re pattern objects are atomic, so this should be cheap
		return cls.re.compile(rf'(?P<pre>(?:([\'"])(?:\\?.)*?\2|[^\'"])*?)(?P<target>{c})(?P<post>.*)')

	@staticmethod
	def find_through_strings(s, c, *, start=0):
		while 1:
			i = s.find(c, start)
			j = s.find('"', start)
			if j == -1 or j > i: return i
			j += 1
			while 1:
				j = s.find('"', j)
				k = j - len(s[:j].rstrip('\\'))
				if k&1 == 0: start = j+1; break
				j += 1


output(r'''
global main
extern printf
extern puts
extern strcmp
extern malloc
extern free
extern exit

segment .text
_0print:
sub rsp, 48

mov rdx, rcx
mov rcx, _p
xor rax, rax
call printf

add rsp, 48
ret

_0puts:
sub rsp, 32

call puts

add rsp, 32
ret

_0alloc:
sub rsp, 32

call malloc

add rsp, 32
ret

_0free:
sub rsp, 32

call free

add rsp, 32
ret

_0println:
sub rsp, 48

mov rdx, rcx
mov rcx, _pln
xor rax, rax
call printf

add rsp, 48
ret

_0printstr:
sub rsp, 48

mov rdx, rcx
mov rcx, _pstr
xor rax, rax
call printf

add rsp, 48
ret

_0printaddr:
sub rsp, 48

mov rdx, rcx
mov rcx, _paddr
xor rax, rax
call printf

add rsp, 48
ret
''')

# Nestless?
# Keep it as simple as possible for now

class Function_header:
	def __init__(
		self, name, typeargs: tuple[str], args: tuple[str, str], ret_type: str,
		tell, line_no
	):
		self.name = name
		self.typeargs = typeargs
		self.args = args
		self.tell = tell
		self.line_no = line_no
		self.instances = {}
		self.ret_type = ret_type

		for arg_entry in args:
			if len(arg_entry) != 2:
				err(f'Invalid syntax {" ".join(arg_entry)!r}. '
					'Expected exactly 2 words for arg entry.')

	def add_sub(self, key: tuple[str], typeargs = ()) -> 'Function_instance':
		# Need to initialise arguments as variables
		if len(key) != len(self.typeargs):
			err(f'Expected {len(self.typeargs)} type arguments '
				f'to {self.name!r}. Got {len(key)}')

		fn_instance = Function_instance(
			self, typeargs, len(self.instances)
		)
		self.instances[key] = fn_instance
		return fn_instance

class Function_instance:
	def __init__(self, template, typeargs: list[str], id):
		self.template = template
		self.id = id
		self.type_mappings = dict(zip(template.typeargs, typeargs))
		self.variables = {}
		self.offset = 0

	def init_args_vars(self, types):
		local_types = types | self.type_mappings
		for typename, argname in self.template.args:
			if argname in self.variables:
				err(f'Multiple instances of argument {argname!r} for function '
					f'{self.template.name!r}')

			# TODO: update variables
			type_list = parse_type(typename, local_types, variables={})
			if type_list is None:
				# TODO: better message
				err(f'Type {typename!r} is not available')
			if len(type_list) != 1:
				err('Type expression must evaluate to exactly one type')
			T, = type_list

			self.offset += T.size
			self.variables[argname] = Variable(
				argname, self.offset, T, self.template.line_no
			)

	def mangle(self):
		return f'_{self.id}{self.template.name}'

class Variable:  # Instantiated when types are known
	def __init__(self, name, offset, type, decl_line_no):
		self.name = name
		self.offset = offset
		self.type = type
		self.decl_line_no = decl_line_no

UNSPECIFIED_TYPE = type('UNSPECIFIED_TYPE', (), {
	'__repr__': lambda s: f'Type<UNSPECIFIED>', 'deref': None
})()
FLAG_TYPE = type('FLAG_TYPE', (), {
	'__repr__': lambda s: f'Type<FLAG>'
})()
MISSING = type('MISSING', (), {
	'__repr__': lambda s: f'<MISSING ARG>'
})()

class Type:
	def __init__(self, name, size = 0, args = ()):
		self.name = name
		self.size = size
		self.fields = {}
		self.deref = None
		self.ref = None
		self.args = args
		self.parent = self

		self.instances = {}
		self.methods = {}

	def __repr__(self):
		return f'{self.__class__.__name__}({self.name})'

	@classmethod
	def get_size(cls, T, *, unspecified = 8):
		if T is UNSPECIFIED_TYPE:
			return unspecified
		if not isinstance(T, cls):
			err(f'[Internal Error] Invalid type {T.__class__} for type')
		return T.size

	def get_instance(self, args: tuple['Type']) -> 'Type':
		global types

		if len(args) != len(self.args):
			err(f'{self} expects {len(self.args)} polymorphic arguments. '
				f'Got {len(args)}.')

		if args in self.instances:
			return self.instances[args]

		if not args:
			self.instances[args] = self
			if not self.fields: return self
			instance = self
			instance.size = 0
			instance_types = types.copy()
		else:
			local_types = dict(zip(self.args, args))
			instance_types = types | local_types

			instance = Type(' '.join(T.name for T in [self, *args]))
			instance.args = args
			instance.parent = self
			instance.methods = self.methods
			self.instances[args] = instance

		for name, field in self.fields.items():
			T = field.type
			if isinstance(T, str):
				# TODO: update variables
				type_list = parse_type(T, instance_types, variables={})
				if len(type_list) != 1:
					err('Declaration requires exactly one type')
				T, = type_list

			print(f'Added field {name!r} of {T} to instance')
			field = Variable(name, instance.size, T, field.decl_line_no)
			instance.fields[name] = field
			instance.size += T.size

		print('INSTANCE OF SIZE', instance.size)
		return instance

	def match_pattern(self, type_str, types) -> dict['type_arg': 'Type']:
		type_queue = [self]
		out_mappings = {}
		for token in type_str.split():

			if not type_queue:
				err(f'Could not match {token!r} from {type_str!r} '
					f'to any type in {self}. '
					'(Note: Type arguments cannot take parameters)')
				# It could work if only the first token is a parameter

			expected_type = type_queue.pop(0)
			if token.startswith('&'):
				if expected_type.deref is None:
					err(f'{expected_type!r} does not have deref for {token!r}')
				expected_type = expected_type.deref
				token = token[1:]
			elif expected_type.deref is not None:
				if token in types:
					print(f'{types[token] = }, {expected_type = }')
					if types[token].deref == expected_type.deref:
						expected_type = types[token]

			if token not in types:
				if token not in out_mappings:
					out_mappings[token] = expected_type
				elif out_mappings[token] is (UNSPECIFIED_TYPE, None):
					out_mappings[token] = expected_type
				elif expected_type is ANY_TYPE:
					out_mappings[token] = expected_type
				elif out_mappings[token] is not expected_type:
					err('Multiple mappings to same argument. '
						f'Trying to map {token!r} to '
						f'{expected_type} and {out_mappings[token]}')
			elif types[token] not in (ANY_TYPE, expected_type.parent):
				# err(f'{expected_type} did not match {token!r} in {type_str!r}')
				err(f'Pattern match of {expected_type} failed for '
					f'{type_str!r} at {token!r}')
			else:
				type_queue.extend(expected_type.args)
			# TODO: append to type_queue

		return out_mappings

	def pointer(self):
		if self.ref is not None: return self.ref
		self.ref = self.__class__('&' + self.name, PTR_SIZE, None)
		self.ref.deref = self
		# print(self.ref, 'has a deref of', self)
		return self.ref

class Branch(Enum):
	ELSE = object()
	WHILE = object()
	WHILEELSE = object()
	FUNCTION = object()

class Exp_type(Enum):
	TOKEN = object()
	GETITEM = object()
	CALL = object()

class Ctrl:
	def __init__(self, ctrl_no, branch: Branch):
		self.ctrl_no = ctrl_no
		self.branch = branch

		if branch is Branch.WHILE:
			self.label = f'_W{ctrl_no}'

	def __repr__(self):
		return f'{self.branch}({self.ctrl_no})'

class Flag(Enum):
	'''
	O is 0 (trigger if the overflow flag is set); NO is 1.
	B, C and NAE are 2 (trigger if the carry flag is set); AE, NB and NC are 3.
	E and Z are 4 (trigger if the zero flag is set); NE and NZ are 5.
	BE and NA are 6 (trigger if either of the carry or zero flags is set); A and NBE are 7.
	S is 8 (trigger if the sign flag is set); NS is 9.
	P and PE are 10 (trigger if the parity flag is set); NP and PO are 11.
	L and NGE are 12 (trigger if exactly one of the sign and overflow flags is set); GE and NL are 13.
	LE and NG are 14 (trigger if either the zero flag is set, or exactly one of the sign and overflow flags is set); G and NLE are 15.
	'''
	o	= 0  # 0
	no	= auto()  # 1
	c = b	= auto()  # 2
	nc = ae	= auto()  # 3
	z = e	= auto()  # 4
	nz = ne	= auto()  # 5
	be	= auto()  # 6
	nbe	= auto()  # 7
	s	= auto()  # 8
	ns	= auto()  # 9
	p	= auto()  # 10
	np	= auto()  # 11
	l	= auto()  # 12
	ge	= auto()  # 13
	le	= auto()  # 14
	g	= auto()  # 15
	ALWAYS	= auto()  # 16
	NEVER	= auto()  # 17

	def __invert__(self):
		return self.__class__(self.value^1)

class In_string(Enum):
	NONE = auto()
	OUT  = auto()
	YES  = auto()
	HEX1 = auto()
	HEX2 = auto()

class Register(Enum):
	a   = auto()
	b   = auto()
	c   = auto()
	d   = auto()
	sp  = auto()
	bp  = auto()
	si  = auto()
	di  = auto()
	r8  = auto()
	r9  = auto()
	r10 = auto()
	r11 = auto()
	r12 = auto()
	r13 = auto()
	r14 = auto()
	r15 = auto()

	def encode(self, *, size):
		# This function is beautiful. I am proud.
		if size not in (1, 2, 4, 8):
			err(f'[Internal error] Invalid Size {size} for register')
		size_log2 = size.bit_length()-1

		if self.name.startswith('r'):
			return self.name + ('b', 'w', 'd', '')[size_log2]
		if size == 1: return self.name + 'l'
		if self.name in 'abcd': out = self.name + 'x'
		else: out = self.name
		return ('', 'e', 'r')[size_log2-1] + out

	def __format__(self, fmt):
		if not fmt.isdigit():
			raise ValueError('Only digits allowed in register format specifier')

		return self.encode(size=int(fmt))

class Address_modes(Enum):
	DEREF = -1
	NONE = 0
	ADDRESS = 1

def size_prefix(size):
	if size == 1:
		return 'byte'
	if size == 2:
		return 'word'
	if size == 4:
		return 'dword'
	if size == 8:
		return 'qword'
	err(f'[Internal error] Invalid size {size} for getting size prefix')

def parse_type(type_str, types, *, variables) -> Optional[list[Type]]:
	# DONE: add pointer support

	if not type_str:
		return []
	if type_str.startswith('&'):
		type_str = type_str[1:].strip()
		pointer = True
	else:
		pointer = False

	match = Patterns.split_word.match(type_str.lstrip())
	print(f'PARSE TYPE {type_str!r} -> {match[1]!r} {match[2]!r}')

	T = match[1]

	if match[2].startswith(':'):
		match = Patterns.split_word.match(match[2][1:].lstrip())

		field = match[1]
		if field == 'type':
			_insts, _clause, exp_type = parse_token(T, types,
				variables=variables)
		else:
			err(f'Unsupported metadata field {field!r} for type')
		T = exp_type
	elif T not in types:
		print(repr(T), 'not in', types)
		# if T == 'V':
		# 	err('Forced')
		return None
	else:
		T = types[T]

	args = parse_type(match[2].lstrip(), types, variables=variables)
	if args is None: return None

	if T.args is None:
		return [T, *args]
	args_len = len(T.args)

	instance = T.get_instance(tuple(args[:args_len]))
	if pointer: instance = instance.pointer()
	if instance is None:
		err(f'Instance is None on {T}.get_instance({args[:args_len]})')
	args[:args_len] = instance,
	return args

	'''
	if T in args:  # If args too big then you're doing something wrong. I can't be bothered to have a hashed copy
		if pointer:
			T = '&' + T
	else:
		T = types[T]
		if pointer:
			T = T.pointer()

	if T is curr_type:
		err(f'Recursive declaration. ({T} within {T})')

	if T is ANY_TYPE:
		err("A variable of type 'any' must be a pointer")
	'''

	err('Types cannot be parsed')
	return [UNSPECIFIED_TYPE]

def parse_token(token: 'stripped', types, *, variables) \
	-> (list[str], Union[str, int], Type):
	# (instructions to get the value of token, expression, type)

	print('Parse token', repr(token))

	clause = None
	val = None
	var = None
	T   = UNSPECIFIED_TYPE
	insts = []
	r_operand = None

	for operator in ('<=', '>=', '<<', '>>', '==', '!=', *'<>+-*/%&^|'):  # big ones first
		operator_idx = Patterns.find_through_strings(token, operator)
		if operator_idx != -1:
			l_operand = token[:operator_idx].strip()
			r_operand = token[operator_idx+len(operator):].strip()

			print(f'{l_operand!r} {operator} {r_operand!r}')
			if not l_operand:
				# err('[Internal error] Unary not checked earlier')
				r_operand = None
				continue

			o_insts, o_clause, o_type = parse_token(r_operand, types,
				variables=variables)
			if o_insts: err(f'Expression {token!r} is too complex')
			token = l_operand
			break

	if token.startswith('&'):
		addr = Address_modes.ADDRESS
		token = token[1:].lstrip()
	elif token.startswith('*'):
		addr = Address_modes.DEREF
		token = token[1:].lstrip()
	else:
		addr = Address_modes.NONE

	colon_idx = Patterns.find_through_strings(token, ':')
	dot_idx = Patterns.find_through_strings(token, '.')

	if colon_idx != -1:
		exp = token[:colon_idx]
		field = token[colon_idx+1:].strip()

		if field == 'size':
			type_list = parse_type(exp, types, variables=fn_instance.variables)
			if type_list is None:
				err(f'Type {exp!r} not available')
			if addr is Address_modes.ADDRESS:
				val = PTR_SIZE
			else:
				if len(type_list) != 1:
					err(f'{exp!r} does not correspond to a single type')
				T, = type_list
				val = T.size
			print(f'{T}:size = {val}')
			T = UNSPECIFIED_TYPE
		elif field == 'type':
			_insts, _clause, exp_type = parse_token(exp, types,
				variables=fn_instance.variables)

			string = bytes(exp_type.name, 'utf-8')
			clause = get_string_label(string, strings)
			T = types['str']

		elif field == 'name':
			type_list = parse_type(exp, types, variables=fn_instance.variables)
			if type_list is None:
				err(f'Type {exp!r} not available')
			if len(type_list) != 1:
				# NOTE: I would 
				err(f'{exp!r} does not correspond to a single type')
			T, = type_list

			string = bytes(T.name, 'utf-8')
			clause = get_string_label(string, strings)
			T = types['str']

		else:  # TODO: type:name, var:type, var:len
			err(f'Unsupported metadata field {field!r} for token')

	elif token.isidentifier():
		# if addr: err("Can't take addresses of local variables yet")
		if token not in variables: err(f'{token!r} not defined')
		var = variables[token]
		offset = var.offset
		clause = f'rsp + {offset}'
		T = var.type
		if addr is Address_modes.ADDRESS:
			T = T.pointer()
			insts.append(f'lea {{dest_reg:{Type.get_size(T)}}}, [{clause}]')
			clause = None
		else:
			clause = f'{size_prefix(var.type.size)} [{clause}]'

	elif dot_idx != -1:
		root = token[:dot_idx]
		chain = token[dot_idx+1:].split('.')
		root = root.strip()
		if root not in variables:
			err(f'{root!r} not defined')

		print(f'Getting a field of {root!r}')
		var = variables[root]
		offset = var.offset
		base_reg = 'rsp'

		T = var.type
		print(f'Getting a field of {root!r} {T}')
		for field in chain:
			field = field.strip()
			print(f'  {field = }')
			if T.deref is not None:
				# We want to dereference T, so we first put it into a register
				size = Type.get_size(T)
				insts += (
					f'mov {{dest_reg:{size}}}, '
					f'{size_prefix(size)} [{base_reg} + {offset}]',
				)
				base_reg = f'{{dest_reg:{size}}}'
				offset = 0
				T = T.deref

			if field not in T.fields: err(f'{T} has no field {field!r}')

			var = T.fields[field]
			for _name, _field in T.fields.items():
				print(' ', _name, _field.type)
			print(f'  {T}.fields[{field!r}]')
			T = var.type
			print(f'  {field = } {T}')
			offset += var.offset

		clause = f'{base_reg} + {offset}'
		if addr is Address_modes.ADDRESS:
			T = T.pointer()
			insts.append(f'lea {{dest_reg:{Type.get_size(T)}}}, [{clause}]')
			clause = None
		else:
			clause = f'{size_prefix(T.size)} [{clause}]'

	elif token.isdigit():
		if addr is Address_modes.ADDRESS:
			err("Can't take address of a integer literal")
		if addr is Address_modes.DEREF:
			err("Can't dereference a integer literal")
		val = int(token)

	elif token.startswith("'"):
		if addr is Address_modes.ADDRESS:
			err("Can't take address of a character literal")
		if addr is Address_modes.DEREF:
			err("Can't dereference a character literal")
		if token[-1] != "'":
			err('Expected end quote (\') at the end of character literal.')

		if token[1] != '\\':
			if len(token) != 3:
				err('Invalid syntax for character literal')
			val = ord(token[1])
		elif token[2] != 'x':
			if len(token) > 4: err('Character literal too long')
			if len(token) < 4: err('Character literal too short')
			c = token[2]
			if   c == '0':  val = 0
			elif c == 't':  val = 9
			elif c == 'n':  val = 10
			elif c == 'e':  val = 27
			elif c == '"':  val = 34
			elif c == "'":  val = 39
			elif c == '\\': val = 92
			else:
				err('Invalid escape sequence')
		else:
			if len(token) != 6:
				err('Invalid syntax for character literal')
			val = int(token[3:5], 16)
		clause = f'{val!r}'
		T = types['char']

	elif token.startswith('"'):
		if addr is Address_modes.ADDRESS: err('Cannot take address of string literal')

		string_data = bytearray()
		h_val = None
		escape = In_string.NONE
		for c in token[1:]:
			if escape is In_string.NONE:
				if   c == '\\': escape = In_string.YES
				elif c == '\"': escape = In_string.OUT
				else: string_data.extend(c.encode())
			elif escape is In_string.YES:
				if c == 'x': escape = In_string.HEX1; continue

				if   c == '0':  string_data.append(0)
				elif c == 't':  string_data.append(9)
				elif c == 'n':  string_data.append(10)
				elif c == 'e':  string_data.append(27)
				elif c == '"':  string_data.append(34)
				elif c == "'":  string_data.append(39)
				elif c == '\\': string_data.append(92)
				else:
					err('Invalid escape sequence')

				escape = In_string.NONE

			elif escape is In_string.HEX1:
				if not c.isdigit() and c.lower() not in 'abcdef':
					err('Invalid hexadecimal escape sequence')
				h_val = int(c, 16) << 8
				escape = In_string.HEX2

			elif escape is In_string.HEX2:
				if not c.isdigit() and c.lower() not in 'abcdef':
					err('Invalid hexadecimal escape sequence')
				string_data.append(h_val | int(c, 16))
				h_val = None
				escape = In_string.NONE

			elif escape is In_string.OUT:
				if not c.isspace():
					err(f'Unexpected character {c!r} after string literal')
		if escape is not In_string.OUT:
			err('Unexpected EOL inside string')

		string = bytes(string_data)
		clause = get_string_label(string, strings)
		T = types['str']

	else:
		err(f'Invalid token syntax {token!r}')

	if val is not None:
		clause = f'{val}'

	if addr is Address_modes.DEREF:
		if clause is not None:
			insts += (
				(f'mov {{dest_reg:{Type.get_size(T)}}}, {clause}',)
			)
		if T.deref in (None, ANY_TYPE):
			err(f'Cannot dereference a value of type {T}')
		size = Type.get_size(T.deref)
		clause = f'{size_prefix(size)} [{{dest_reg:{Type.get_size(T)}}}]'
		T = T.deref

	if r_operand is not None:
		insts = [
			# *o_insts,  # too complex if not empty
			# *insts,  # always empty. Actually, no:
			*insts,

			*(() if clause is None else
				(f'mov {{dest_reg:{Type.get_size(T)}}}, {clause}',)),

			*get_operator_insts(operator, o_clause, o_type)
		]
		clause = None
		print(f'OPERATOR {operator!r} using {T} and {o_type} ({addr = }) gives... ', end='')
		T = operator_result_type(operator, T, o_type)
		print(T)
	return insts, clause, T

# muddles rbx if dest_reg is Flag
def parse_exp(exp: 'stripped', *, dest_reg, fn_queue, variables) -> Type:
	# extract call_function(fn, args)

	global types, fn_types

	idx = Patterns.find_through_strings(exp, '(')
	if idx == -1:
		idx = Patterns.find_through_strings(exp, '[')
		if idx == -1:
			exp_type = Exp_type.TOKEN
		else:
			exp_type = Exp_type.GETITEM
	else:
		exp_type = Exp_type.CALL

	if exp_type is Exp_type.TOKEN:
		# [x] T flag, dest_reg flag -> return T

		insts, exp_clause, T = parse_token(exp, fn_types, variables=variables)
		# print('Token', repr(exp), 'has a type of', T)

		if isinstance(T, Flag):
			print('Parsed flag token', exp, '->', T)
			for inst in insts:
				output(inst.format(dest_reg=Register.b))
				# err('[Internal error] Multiple instructions from a flag token')

			if dest_reg is Flag: return T

			if T.value is Flag.ALWAYS:  exp_clause = '1'
			elif T.value is Flag.NEVER: exp_clause = '0'

			# [x] T flag, dest_reg val -> setcc; ALWAYS/NEVER converted to values
			else:
				output(f'set{T.name} {dest_reg.encode(size=8)}')
				return UNSPECIFIED_TYPE


			T = UNSPECIFIED_TYPE


		size = Type.get_size(T)


		# [x] T val, dest_reg val -> mov
		if dest_reg is not Flag:
			for inst in insts:
				output(inst.format(dest_reg=dest_reg))
			if exp_clause is not None:
				# print('Moving into rax', T, size)
				output(f'mov {dest_reg:{size}}, {exp_clause.format(dest_reg=dest_reg)}')

			return T
		else:
			# [ ] T val, dest_reg flag -> test

			for inst in insts:
				output(inst.format(dest_reg=Register.a))

			if T is UNSPECIFIED_TYPE and exp_clause.isdigit():
				val = int(exp_clause)
				if val: return Flag.ALWAYS
				else: return Flag.NEVER

			elif T is types['str']:
				# TODO: account for null strings
				return Flag.ALWAYS
			elif exp_clause is not None:
				# works only if exp_clause can be a dest
				output(f'test {exp_clause}, -1')
				return Flag.nz
			else:
				output(f'test {{0:{T.size}}}, {{0:{T.size}}}'
					.format(Register.a))
				return Flag.nz

	print('Parse exp', repr(exp))

	if exp_type is Exp_type.CALL and exp[-1] != ')':
		err("Expected function call to end with a closing parenthesis ')'")
	if exp_type is Exp_type.GETITEM and exp[-1] != ']':
		err("Expected item access to end with a closing square bracket ']'")


	if exp_type is Exp_type.GETITEM:
		# TODO: method call support
		if exp.startswith('&'):
			fn_name = '_getref'
			exp = exp[1:]
			idx -= 1
		else:
			fn_name = '_getitem'

		insts, clause, T = parse_token(exp[:idx].strip(), fn_types, variables=variables)
		print(f'Type of {exp[:idx]!r} is {T}')

		# TODO: What if I want an integer of a different type?

		for inst in insts:
			output(inst.format(dest_reg=Register.c))
		if clause is not None:
			reg_str = Register.c.encode(size=Type.get_size(T))
			output(f'mov {reg_str}, {clause.format(dest_reg=Register.c)}')

		arg_types = [T]
		fn_name = f'{T.name}.{fn_name}'
	else:
		fn_name = exp[:idx].strip()
		if fn_name.startswith('*'):
			fn_deref = True
			fn_name = fn_name[1:].strip()
		else:
			fn_deref = False
		arg_types = []

	ret_type = call_function(fn_name, arg_types, exp[idx:], variables=variables)

	print(f'{dest_reg = }')
	if dest_reg is Flag:
		# output is in rax
		print('Classified as a flag')
		output(f'test {Register.a:{ret_type.size}}, '
			f'{Register.a:{ret_type.size}}')
		return Flag.nz

	print('Not classified as a flag')

	return ret_type


# args_str must include the starting bracket
# args_str = '(arg1, arg2)', but not 'arg, arg2)'
def call_function(fn_name, arg_types, args_str, *, variables):
	if fn_name == 'alloc':
		alloc_type = None
		alloc_fac  = None

	idx = 0
	while idx != -1:
		if len(arg_types) >= len(arg_regs):
			err('Only upto 4 arguments are allowed')
		arg_reg = arg_regs[len(arg_types)]

		end = Patterns.find_through_strings(args_str, ',', start=idx+1)
		arg = args_str[idx+1:end].strip()

		idx = end
		if not arg and idx == -1: break

		if fn_name == 'alloc':
			# doesn't work if only one argument is provided. Should work.
			if alloc_type is None:
				type_list = parse_type(arg, fn_types, variables=variables)


				if type_list is None:
					err(f'Type not found in {arg}')
				if len(type_list) != 1:
					err('Expected exactly one type in alloc()')

				alloc_type, = type_list
				# if alloc_type is ANY_TYPE:
				# 	err(f'{alloc_type} has no associated size')
				alloc_fac = alloc_type.size
				continue

			arg = f'{alloc_fac}*{arg}'

		print('Arg:', arg)

		insts, clause, T = parse_token(arg, fn_types, variables=variables)
		print(f'Type of {arg!r} is {T}')

		# TODO: What if I want an integer of a different type?

		for inst in insts:
			output(inst.format(dest_reg=arg_reg))
		if clause is not None:
			reg_str = arg_reg.encode(size=Type.get_size(T))
			output(f'mov {reg_str}, {clause.format(dest_reg=arg_reg)}')

		arg_types.append(T)

	if fn_name == 'alloc' and not arg_types:
		arg_types.append(UNSPECIFIED_TYPE)
		output(f'mov {arg_reg:8}, {alloc_fac}')

	caller_type_name, dot, fn_name = fn_name.rpartition('.')

	if dot:
		type_list = parse_type(caller_type_name, fn_types, variables=variables)
		if type_list is None:
			err(f'Type {caller_type_name!r} is not available')
		if len(type_list) != 1:
			err('Method calls expect exactly one type')
		caller_type, = type_list
		# if caller_type.deref is not None:
		# 	caller_type = caller_type.deref

		# NOTE: Temp
		if fn_name == 'hash':  # fn_name in default_methods
			if dest_reg is Flag:
				output('test rcx, rcx')
				return Flags.nz
			output('mov rax, rcx')
			return types['int']
		elif fn_name not in caller_type.methods:
			# check for deref only if method doesn't exist
			if caller_type.deref is not None:
				caller_type = caller_type.deref
				if fn_name not in caller_type.methods:
					err(f'{caller_type} has no method named {fn_name!r}')
			else:
				err(f'{caller_type} has no method named {fn_name!r}')
		fn_header = caller_type.methods[fn_name]
		print('METHOD  ', fn_name, fn_header)
	elif fn_name not in function_headers:
		err(f'No function named {fn_name!r}')
	else:
		fn_header = function_headers[fn_name]
		print('FUNCTION', fn_name, fn_header)
		caller_type = None

	# use fn_header.typeargs, fn_header.args
	# We could store {typename: Type}, but rn we have {typename: typename}

	if len(arg_types) != len(fn_header.args):
		fl = len(fn_header.args)
		al = len(arg_types)

		err(f'{fn_header.name!r} expects exactly '
			f'{fl} argument{"s" * (fl != 1)}, '
			f'but {al} {"were" if al != 1 else "was"} provided')

	# Populate type_mappings
	type_mappings = {T: None for T in fn_header.typeargs}
	if caller_type is not None:
		type_mappings |= dict(
			zip(caller_type.parent.args, caller_type.args)
		)
	print('TYPE MAPPING USING', fn_header.args, 'AND', arg_types)
	for i, ((type_str, arg_name), arg_type) in enumerate(zip(fn_header.args, arg_types), 1):
		if arg_type is not UNSPECIFIED_TYPE:
			curr_mappings = arg_type.match_pattern(type_str, types)
		elif parse_type(type_str, types, variables=variables) is None:
			# don't update mappings
			if len(type_str.split(maxsplit=1)) > 1: continue
			# We have to expect UNSPECIFIED_TYPE in the for loop
			curr_mappings = {type_str.lstrip(): UNSPECIFIED_TYPE}
		else: continue  # parse_type is not None, so it won't change

		for type_arg, matched_type in curr_mappings.items():
			if type_arg not in type_mappings:
				print(f'{type_mappings = }')
				err(f'{type_arg!r} in {type_str} is neither '
					'an existing type nor a type argument')
			elif type_mappings[type_arg] in (UNSPECIFIED_TYPE, None):
				type_mappings[type_arg] = matched_type
			elif matched_type is ANY_TYPE:
				type_mappings[type_arg] = matched_type
			elif matched_type not in (type_mappings[type_arg], UNSPECIFIED_TYPE):
				err('Multiple mappings to same argument. '
					f'Trying to map {type_arg!r} to '
					f'{matched_type} and {type_mappings[type_arg]} '
					f'{arg_types}')

	# Handles UNSPECIFIED_TYPE
	for typename, subbed_type in type_mappings.items():
		if subbed_type is UNSPECIFIED_TYPE:
			type_mappings[typename] = types['int']

	try:
		instance_key = tuple(type_mappings[typearg_name].name for typearg_name in fn_header.typeargs)
	except AttributeError:
		for typearg_name in fn_header.typeargs:
			if type_mappings[typearg_name] is not None: continue
			err(f'Type argument {typearg_name!r} not mapped in {fn_header.name!r}')
		err('[Internal Error] All types mapped but still got a TypeError')

	if instance_key in fn_header.instances:
		fn_instance = fn_header.instances[instance_key]
	else:
		fn_queue.append((fn_header, instance_key))
		fn_instance = fn_header.add_sub(
			instance_key, (*type_mappings.values(),)
		)

	output('call', fn_instance.mangle())

	if fn_name == 'alloc':
		return alloc_type.pointer()

	# print(f'{fn_header.ret_type = }')

	ret_type_list = parse_type(fn_header.ret_type, types | type_mappings,
		variables = variables)
	if ret_type_list is None:
		# TODO: better message
		err(f'No such type {fn_header.ret_type}')
	elif len(ret_type_list) != 1:
		err('Return type string does not evaluate to a single type')
	ret_type, = ret_type_list

	return ret_type

def get_operator_insts(operator, operand_clause, operand_type):
	# Should not muddle registers. So wee can't call functions.

	# I don't put any constraints. Makes it unsafe, but also flexible.
	# We'll see if that's a good idea.

	if   operator == '+': inst = 'add'
	elif operator == '-': inst = 'sub'
	elif operator == '&': inst = 'and'
	elif operator == '^': inst = 'xor'
	elif operator == '|': inst = 'or'
	elif operator in ('<', '>', '<=', '>='): inst = 'cmp'
	elif operator in ('==', '!='):
		# get_operator_insts() should take type of both operands
		if operand_type is types['str']:
			return [
				f'mov rcx, {{dest_reg:8}}',
				f'mov rdx, {operand_clause}',
				'call strcmp',
				'cmp al, 0'
			]
		inst = 'cmp'
	elif operator == '<<':
		size = Type.get_size(operand_type)
		return [
			f'mov {Register.c:{size}}, {operand_clause}',
			f'shl {{dest_reg:{size}}}, cl',
		]
	elif operator == '>>':
		size = Type.get_size(operand_type)
		return [
			f'mov {Register.c:{size}}, {operand_clause}',
			f'shr {{dest_reg:{size}}}, cl',
		]
	elif operator == '*':
		size = Type.get_size(operand_type)
		return [
			f'mov {Register.a:{size}}, {{dest_reg:{size}}}',
			f'xor rdx, rdx',
			f'mov {Register.b:{size}}, {operand_clause}',
			f'mul {Register.b:{size}}',
			f'mov {{dest_reg:{size}}}, {Register.a:{size}}'
		]
	elif operator == '/':
		size = Type.get_size(operand_type)
		return [
			f'mov {Register.a:{size}}, {{dest_reg:{size}}}',
			f'xor rdx, rdx',
			f'mov {Register.b:{size}}, {operand_clause}',
			f'div {Register.b:{size}}',
			f'mov {{dest_reg:{size}}}, {Register.a:{size}}'
		]
	elif operator == '%':
		size = Type.get_size(operand_type)
		return [
			f'mov {Register.a:{size}}, {{dest_reg:{size}}}',
			f'xor rdx, rdx',
			f'mov {Register.b:{size}}, {operand_clause}',
			f'div {Register.b:{size}}',
			f'mov {{dest_reg:{size}}}, {Register.d:{size}}'
		]
	else:
		# others?
		err(f'Operator {operator} not supported')

	return [f'{inst} {{dest_reg:{Type.get_size(operand_type)}}}, {operand_clause}']

def operator_result_type(operator, l_type, r_type) -> Type:
	# str  + int
	# str  + char
	# int  + int
	# int  + char
	# char + char

	# str  - int
	# str  - char
	# int  - int
	# int  - char
	# char - char
	if types['void'] in [l_type, r_type]:
		err(f'Unsupported operator {operator!r} between '
			f'{l_type} and {r_type}')

	if types['str'] in [l_type, r_type]:
		if r_type is l_type or UNSPECIFIED_TYPE in [l_type, r_type]:
			if operator == '==': return Flag.e
			if operator == '!=': return Flag.ne

		if r_type is l_type or operator not in ('+', '-'):
			err(f'Unsupported operator {operator!r} between '
				f'{l_type} and {r_type}')
		return types['str']

	if operator in '+-':
		if l_type.deref is not None and r_type.deref is None:
			if r_type not in (types['int'], UNSPECIFIED_TYPE):
				err(f'Cannot offset a pointer using {r_type}')
			return l_type

	if operator == '+':
		if r_type.deref is not None and l_type.deref is None:
			if l_type not in (types['int'], UNSPECIFIED_TYPE):
				err(f'Cannot offset a pointer using {l_type}')
			return r_type

		if None not in (l_type.deref, r_type.deref):
			err('Cannot add pointers')

	# account for custom types
	if operator == '==': return Flag.e
	if operator == '!=': return Flag.ne

	if l_type not in builtin_types:
		err(f'Cannot use operator {operator!r} on custom type {l_type}')
	if r_type not in builtin_types:
		err(f'Cannot use operator {operator!r} on custom type {r_type}')

	if operator == '>':  return Flag.g
	if operator == '<':  return Flag.l
	if operator == '>=': return Flag.ge
	if operator == '<=': return Flag.le

	if UNSPECIFIED_TYPE in (l_type, r_type):
		return UNSPECIFIED_TYPE

	if types['int'] in (l_type, r_type):
		return types['int']

	return types['char']

def get_string_label(string, strings):
	if string in strings: return strings[string]
	label = f'_s{len(strings)}'
	strings[string] = label
	return label

# Strongly typed

# First pass, get the declarations

fn_queue = []
# Builtins
types = {
	'str': Type('str', 8),
	'int': Type('int', 4),
	'char': Type('char', 1),
	'void': Type('void', 0),
	'any': Type('any', None),
}
ANY_TYPE = types['any']
types['str'].deref = types['char']

builtin_types = {*types.copy().values(), UNSPECIFIED_TYPE, FLAG_TYPE}

# Builtins
# Function_header(name, type_args, *args(type, name), ret_type, tell, line_no)
function_headers = {
	'print': Function_header('print', (), (('int', 'n'),), 'void', 0, 0),
	'println': Function_header('println', (), (('int', 'n'),), 'void', 0, 0),
	'printstr': Function_header('printstr', (), (('str', 's'),), 'void', 0, 0),
	'printaddr': Function_header('printaddr', (), (('&any', 'p'),), 'void', 0, 0),
	'puts': Function_header('puts', (), (('str', 's'),), 'void', 0, 0),
	'alloc': Function_header('alloc', (), (('int', 'n'),), '', 0, 0),
	'free': Function_header('free', (), (('&any', 'p'),), 'void', 0, 0),
}

for fn in function_headers.values():
	fn.add_sub(())

for builtin_type in types.values():
	for method in builtin_type.methods.values():
		method.add_sub(())

arg_regs = (Register.c, Register.d, Register.r8, Register.r9)

in_function = False
curr_type = None
scope_level = 0

tell = 0
for Shared.line_no, Shared.line, in enumerate(Shared.infile, 1):
	tell += len(Shared.line) + crlf
	match = Patterns.stmt.match(Shared.line)
	line = match[2]

	match = Patterns.split_word.match(line)

	if curr_type is not None and not in_function:
		if match[1] == 'type':
			err('Recursive types are not supported')

		if match[1] == 'let':
			match = Patterns.split_word.match(match[2])
			name = match[1]
			T = match[2]

			if name in curr_type.fields:
				var = curr_type.fields[name]
				err(f'Field {name!r} already declared in '
					f'line {var.decl_line_no}')

			curr_type.fields[name] = (
				Variable(name, curr_type.size, T, Shared.line_no)
			)
			print(f'  Created a field {name!r} of {T}')
			if curr_type.size is not None:
				if not isinstance(T, Type):
					curr_type.size = None
				else:
					curr_type.size += T.size

		elif match[1] == 'fn':  # methods
			# name typeargs : type arg, comma separated
			pre, arrow, ret_type = match[2].partition('->')
			pre, _, args = pre.partition(':')
			name, *typeargs = pre.split()
			args = args.strip()
			args = args.split(',')
			if args[-1] == '': args.pop()

			if not arrow: ret_type = 'void'
			else: ret_type = ret_type.strip()

			# print(repr(line), name, typeargs, args, ret_type, sep = ',\t')

			if name in curr_type.methods:
				err(f'Function {name!r} already defined.')

			fn = Function_header(f'{curr_type.name}.{name}',
				(*typeargs, *curr_type.args),  # curr_type.args may have Type objects?
				tuple(arg.rsplit(maxsplit=1) for arg in args),
				ret_type, tell, Shared.line_no
			)
			print(f'NEW FUNCTION HEADER {name}: {fn.args = }')
			curr_type.methods[name] = fn

			in_function = True
			scope_level += 1

		elif match[1] == 'end':
			print('After definition:')
			for name, field in curr_type.fields.items():
				print(name, field.type)
			curr_type = None
			scope_level -= 1

	elif not match: continue
	elif match[1] == 'type':
		if scope_level: err('Local type definitions are not yet supported')

		name, *args = match[2].split()
		# if args: err('Polymorphic types are not yet supported')

		if name in types: err(f'Type {name!r} already defined')
		curr_type = Type(name, args = args)
		types[name] = curr_type
		print('NEW TYPE', curr_type)

		scope_level += 1

		# if T in args:  # If args too big then you're doing something wrong. I can't be bothered to have a hashed copy
		# 	if pointer:
		# 		T = '&' + T
		# else:
		# 	T = types[T]
		# 	if pointer:
		# 		T = T.pointer()

		# if T is curr_type:
		# 	err(f'Recursive declaration. ({T} within {T})')

		# if T is ANY_TYPE:
		# 	err("A variable of type 'any' must be a pointer")

	elif match[1] == 'fn':  # function header
		if scope_level:
			err('Local functions are not supported')

		# name typeargs : type arg, comma separated
		pre, arrow, ret_type = match[2].partition('->')
		pre, _, args = pre.partition(':')
		name, *typeargs = pre.split()
		args = args.strip()
		args = args.split(',')
		if args[-1] == '': args.pop()

		if not arrow: ret_type = 'void'
		else: ret_type = ret_type.strip()

		# print(repr(line), name, typeargs, args, ret_type, sep = ',\t')

		if name in function_headers:
			err(f'Function {name!r} already defined.')

		fn = Function_header(
			name, (*typeargs,), tuple(arg.rsplit(maxsplit=1) for arg in args),
			ret_type, tell, Shared.line_no
		)
		print(f'NEW FUNCTION HEADER {name}: {fn.args = }')
		function_headers[name] = fn

		if name == 'main':  # not be confused with if __name__ == '__main__':
			fn.add_sub(())
			fn_queue.append((fn, ()))

		in_function = True
		scope_level += 1

	elif match[1] in ('if', 'while'):
		scope_level += 1

	elif match[1] == 'end':
		scope_level -= 1
		if scope_level < 0:
			err('end statement does not match any block')
		elif scope_level == 1 and in_function and curr_type is not None:
			in_function = False
		elif scope_level == 0:
			in_function = False

if not fn_queue:
	err("No definition of function 'main' found.")

strings = {}
ctrl_no = 0

while fn_queue:
	fn, instance_key = fn_queue.pop(0)
	fn_instance = fn.instances[instance_key]
	# fn_instance = fn.add_sub(instance_key)
	# if fn_instance is None: continue

	# 2 passes. allocate variable space first

	output(f'\n; {fn_instance.type_mappings}')
	fn_types = types | fn_instance.type_mappings

	fn_instance.init_args_vars(types)
	offset = fn_instance.offset

	scope_level = 1

	Shared.infile.seek(fn.tell)
	for Shared.line_no, Shared.line in enumerate(Shared.infile, fn.line_no+1):
		# let x type
		# Variable{name, offset, type}
		# Type{name, size, fields}
		# Field{name, offset, type}

		line = Patterns.stmt.match(Shared.line)[2]
		if not line: continue

		match = Patterns.split_word.match(line)
		if match[1] == 'let':
			name, type_str = match[2].split(maxsplit=1)

			if name in fn_instance.variables:
				var = fn_instance.variables[name]
				err(f'Variable {name!r} already declared in '
					f'line {var.decl_line_no}')

			type_list = parse_type(type_str.lstrip(), fn_types,
				variables=fn_instance.variables)
			if type_list is None:
				err(f'A type in {type_str!r} is not defined')
				# one of the arguments in T is polymorphic
				# DRY this up maybe
				# match = Patterns.split_word.match(type_str)
				# if match[1].startswith('&'):
				# 	pointer = True
				# 	T = fn_types[match[1][1:].strip()]
				# else:
				# 	pointer = False
				# 	T = fn_types[match[1]]
				# T.get_instance(tuple(parse_type(match[2])))

			print(f'Declaration type list of {type_str!r}: {type_list}')
			T, = type_list

			if T is ANY_TYPE:
				err("A variable of type 'any' must be a pointer")

			offset += T.size
			fn_instance.variables[name] = (
				Variable(name, offset, T, Shared.line_no)
			)

		elif match[1] in ('if', 'while'):
			scope_level += 1

		elif match[1] == 'end':
			scope_level -= 1
			if not scope_level: break  # end of function


	# I might be able to support overloading too and just disallow conflicts
	# We already have the line number in the struct, so we can error nicely

	# Code gen
	if fn.name == 'main':
		output('main:')
		output('push rbp')
		output('mov rbp, rsp')
		output()
	else:
		output(f'{fn_instance.mangle()}:')
	output(f'mov rbp, rsp')  # 32 extra bytes are always required

	# align and push only if there are function calls
	offset = ((offset+1) | 15) + 33  # (round up to multiple of 15) + 32
	output(f'sub rsp, {offset}')
	output('push rbp')

	for var in fn_instance.variables.values():
		var.offset = offset - var.offset

	# Populate arguments
	if len(fn.args) > len(arg_regs):
		err('[Internal Error] Too many arguments; this was not checked earlier')
	for (_, argname), arg_reg in zip(fn.args, arg_regs):
		arg = fn_instance.variables[argname]
		reg_str = arg_reg.encode(size=arg.type.size)
		output(f'mov {size_prefix(arg.type.size)} [rsp + {arg.offset}], ',
			reg_str)

	ctrl_stack = [Ctrl(0, Branch.FUNCTION)]

	print('\n', fn.name, instance_key, sep = '')

	Shared.infile.seek(fn.tell)
	for Shared.line_no, Shared.line in enumerate(Shared.infile, fn.line_no+1):

		match = Patterns.stmt.match(Shared.line)
		line = match[2]  # maybe indentation?
		if not line: continue

		output(f'; ({Shared.line_no}) {Shared.line.strip()}')
		print(f'{Shared.line_no} {Shared.line.strip()!r}')

		match = Patterns.split_word.match(line)

		if not match: match = Subscriptable(); print(match)

		if   match[1] == 'let': continue
		elif match[1] == 'return':
			if fn.name == 'main': dest_reg = Register.c
			else: dest_reg = Register.a

			if not match[2]:
				ret_type = types['void']
			else:
				# We don't use the expected size
				# for the case of returning UNSPECIFIED_TYPE
				ret_type = parse_exp(match[2].strip(),
					dest_reg = dest_reg, fn_queue = fn_queue,
					variables = fn_instance.variables)

			fn_type_list = parse_type(fn.ret_type, fn_types,
				variables=fn_instance.variables)

			if fn_type_list is None:
				# TODO: better message
				err(f'Type {fn.ret_type!r} is not available.')
			if len(fn_type_list) != 1:
				err('Return type must be exactly one type')
			expected_ret_type, = fn_type_list

			if ret_type not in (expected_ret_type, UNSPECIFIED_TYPE):
				err('Mismatched type. '
					f'{fn.name} expects {fn_types[fn.ret_type]}. '
					f'Trying to return {ret_type}')

			if fn.name == 'main':
				output('call exit')
			else:
				output('pop rbp')
				output('mov rsp, rbp')
				output('ret')

		elif match[1] == 'while':
			ctrl = Ctrl(ctrl_no, Branch.WHILE)
			output(f'{ctrl.label}:')
			ctrl_stack.append(ctrl)

			# TODO: parse_exp() returns a Flag object if dest_reg is flags
			ret_flag = parse_exp(match[2].strip(),
				dest_reg = Flag, fn_queue = fn_queue,
				variables = fn_instance.variables
			)



			if ret_flag is Flag.ALWAYS:
				output(f'jmp _E{ctrl_no}')
			elif ret_flag is not Flag.NEVER:
				output(f'j{(~ret_flag).name} _E{ctrl_no}')

			ctrl_no += 1

		elif match[1] == 'if':
			ctrl = Ctrl(ctrl_no, 0)
			ctrl_stack.append(ctrl)

			# TODO: parse_exp() returns a Flag object if dest_reg is flags
			ret_flag = parse_exp(match[2].strip(),
				dest_reg = Flag, fn_queue = fn_queue,
				variables = fn_instance.variables
			)

			if ret_flag is Flag.ALWAYS:
				output(f'jmp _E{ctrl_no}_1')
			elif ret_flag is not Flag.NEVER:
				output(f'j{(~ret_flag).name} _E{ctrl_no}_1')

			ctrl_no += 1

		elif match[1] == 'elif':
			if not ctrl_stack or ctrl_stack[-1].branch is Branch.WHILE:
				err('elif is not after if')

			ctrl = ctrl_stack[-1]

			ctrl.branch += 1

			output(f'jmp _END{ctrl.ctrl_no}')
			output(f'_E{ctrl.ctrl_no}_{ctrl.branch}:')


			# TODO: parse_exp() returns a Flag object if dest_reg is flags
			ret_flag = parse_exp(match[2].strip(),
				dest_reg = Flag, fn_queue = fn_queue,
				variables = fn_instance.variables
			)

			if ret_flag is Flag.ALWAYS:
				output(f'jmp _E{ctrl.ctrl_no}_{ctrl.branch+1}')
			elif ret_flag is not Flag.NEVER:
				output(f'j{(~ret_flag).name} _E{ctrl.ctrl_no}_{ctrl.branch+1}')

		elif match[1] == 'else':
			if not ctrl_stack or ctrl_stack[-1].branch is Branch.WHILE:
				err('else is not after if')

			ctrl = ctrl_stack[-1]

			output(f'jmp _END{ctrl.ctrl_no}')
			output(f'_E{ctrl.ctrl_no}_{ctrl.branch+1}:')

			ctrl.branch = Branch.ELSE

		elif match[1] == 'end':
			if not ctrl_stack: err("'end' outside any control block.")

			ctrl = ctrl_stack.pop()
			if ctrl.branch is Branch.WHILE:
				output(f'jmp {ctrl.label}')
				output(f'_E{ctrl.ctrl_no}:')
			elif ctrl.branch is Branch.ELSE:
				output(f'_END{ctrl.ctrl_no}:')
			elif ctrl.branch is Branch.FUNCTION:
				break
			else:  # ctrl.branch is an integer
				output(f'_E{ctrl.ctrl_no}_{ctrl.branch+1}:')
				output(f'_END{ctrl.ctrl_no}:')

		elif match[1] == 'break':
			for ctrl in reversed(ctrl_stack):
				if ctrl.branch is Branch.WHILE:
					break
			else:
				err('break outside a loop')

			output(f'jmp _E{ctrl.ctrl_no}')

		elif match[1] == 'continue':
			for ctrl in reversed(ctrl_stack):
				if ctrl.branch is Branch.WHILE:
					break
			else:
				err('continue outside a loop')

			output(f'jmp {ctrl.label}')

		elif match[1] == 'type':
			err('Local types are not yet supported')

		else:
			match = Patterns.through_strings(r'(?<!=)=(?!=)').match(line)

			if match:
				exp = match['post'].strip()
				dest = match['pre'].strip()
				print(f'{exp = }; {dest = }; {match[2] = }')
			else:
				exp = line
				dest = None


			ret_type = parse_exp(exp.strip(),
				dest_reg = Register.a, fn_queue = fn_queue,
				variables = fn_instance.variables)

			if dest is not None:
				index = Patterns.find_through_strings(dest, '[')

				if index != -1:
					dest_token = dest[:index]
					args_str = dest[index:].strip()
				else:
					dest_token = dest

				insts, dest_clause, dest_type = parse_token(dest_token, fn_types,
					variables = fn_instance.variables)

				if not dest_clause: err('Destination too complex')

				if dest_type is UNSPECIFIED_TYPE:
					err(f'Cannot assign to {dest}')

				if index != -1:
					first_arg = arg_regs[0]
					second_arg = arg_regs[1]

					# the value of the expression is in rax
					for inst in insts:
						output(inst.format(dest_reg=first_arg))

					print(f'{dest_clause = }')
					dest_clause = dest_clause.format(dest_reg=first_arg)
					output(f'mov {first_arg:{dest_type.size}}, {dest_clause}')
					# print(f'Moving into {dest_clause!r}')
					ret_size = Type.get_size(ret_type)
					output(f'mov {second_arg:{ret_size}}, '
						f'{Register.a:{ret_size}}')

					setitem_result = call_function(f'{dest_type.name}._setitem',
						[dest_type, ret_type], args_str,
						variables=fn_instance.variables)

					if setitem_result is not types['void']:
						err('')

				else:
					if ret_type not in (UNSPECIFIED_TYPE, dest_type):
						err(f'Cannot assign {ret_type} into '
							f'variable {dest} of {dest_type}')

					# the value of the expression is in rax
					for inst in insts:
						output(inst.format(dest_reg=Register.b))

					dest_clause = dest_clause.format(dest_reg=Register.b)
					# print(f'Moving into {dest_clause!r}')
					output(f'mov {dest_clause}, {Register.a:{dest_type.size}}')


			output()

	# which registers need to be preserved now?
	if fn.name == 'main':
		output('xor rcx, rcx')
		output('call exit')
	else:
		output('pop rbp')
		output('mov rsp, rbp')
		output('ret')

output(r'''
segment .data
_p: db `%d`, 0
_pstr: db `%s`, 0
_pln: db `%d\n`, 0
_paddr: db `%p\n`, 0
''')

for string, label in strings.items():
	encoded_string = repr(string)[2:-1].replace('`', '\\`')
	output(f'{label}: db `{encoded_string}`, 0')
