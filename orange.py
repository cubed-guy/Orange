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
# DONE: array deref syntax
# DONE: array assignment syntax
# DONE: extern
# DONE: nested typedef
# DONE: module imports (modules are types)
# DONE: arrays, dictionaries, std
# DONE: constants
# DONE: big moves
# DONE: enums
# DONE: import extern
# DONE: rename UNSPECIFIED_TYPE -> UNSPECIFIED_INT
# DONE: exposed renaming support
# DONE: explicit enum field ids
# DONE: Big enum variant checks
# TODO: correct operator types with UNSPECIFIED_INT
# TODO: SDL bindings
# TODO: enum consts
# TODO: const operations
# TODO: check non-void return
# TODO: hex and bin literals
# TODO: more unaries
# TODO: namespace aliasing
# TODO: big args
# TODO: stack arguments
# TODO: a (better) way to cast variables, Exposed T, Self, :exposed
# TODO: inline (and other?) optimisations
# TODO: SoA support
# TODO: conditional compilation

from sys import argv
from enum import Enum, auto
from enum import Flag as Enum_flag
from typing import Union, Optional
from os import system, getcwd
import os.path

class Shared:
	debug = True
	assemble = True
	link = True
	arch = None
	line = '[DEBUG] ** Empty line **'
	line_no = 0
	library_set = set()
	libraries = []
	library_paths = set()
	imports = {}

class Subscriptable:
	def __getitem__(self, key):
		return f'{self.__class__.__name__}_{id(self)&0xffff:x04}[{key}]'

class Arch(Enum):
	elf64 = auto()
	win64 = auto()

if __name__ != '__main__': Shared.debug = True
elif '-d' in argv: Shared.debug = True; argv.remove('-d')
else: Shared.debug = False
Shared.debug = True

if '-e' in argv:
	Shared.assemble = False
	Shared.link = False
	argv.remove('-e')
if '-a' in argv:
	Shared.link = False
	argv.remove('-a')

if   '-win' in argv: Shared.arch = Arch.win64; argv.remove('-win')
elif '-elf' in argv: Shared.arch = Arch.elf64; argv.remove('-elf')
elif Shared.debug: Shared.arch = Arch.win64
else:
	# print('Format not specified. A "-win" or "-elf" flag is required.')
	quit(1)

if len(argv) <2:
	if Shared.debug: argv.append('example.or')
	else: print('Input file not specified'); quit(1)
file_name = argv[1].rpartition('.')[0]
if len(argv)<3: argv.append(file_name+'.asm')

PTR_SIZE = 8
crlf = int(Shared.arch is Arch.win64)

arg_infile = open(argv[1])
Shared.out = open(argv[2], 'w')
def output(*args, file = Shared.out, **kwargs):
	if None in args:
		err('[Internal Error] None passed into output()')
	print(*args, **kwargs, file = file)

def err(msg):
	print(f'File "{Shared.infile.name}", line {Shared.line_no}')
	print('   ', Shared.line.strip())
	if Shared.debug: raise RuntimeError(msg)

	print(msg)
	quit(1)

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
	def alias_through_strings(s, *, start=0) -> tuple[str, str]:
		while 1:
			i = s.find('as ', start)

			j = s.find('"', start, i)
			if j == -1:
				if i == -1: return None, s

				if i != 0 and not s[i-1].isalpha():
					return s[:i].rstrip(), s[i+2:].lstrip()

				start = i+1
				continue

			j += 1
			while 1:
				j = s.find('"', j)
				k = j - len(s[:j].rstrip('\\'))
				if not k&1: start = j+1; break
				j += 1

	@staticmethod
	def find_through_strings(s, c, *, start=0):
		while 1:
			i = s.find(c, start)
			j = s.find('"', start, i)
			if j == -1: return i
			j += 1
			while 1:
				j = s.find('"', j)
				k = j - len(s[:j].rstrip('\\'))
				if not k&1: start = j+1; break
				j += 1

	@staticmethod
	def rfind_through_strings(s, c, *, start=None):
		while 1:
			i = s.rfind(c, 0, start)
			j = s.rfind('"', i, start)
			if j == -1: return i
			j -= 1
			while 1:
				j = s.rfind('"', 0, j)
				l = len(s[:j].rstrip('\\'))
				slashes = j - l
				if not slashes&1: start = l; break
				j -= 1

class ParseTypeError(ValueError):
	pass

output(r'''
extern strcmp
extern malloc
extern exit

segment .text
_0alloc:
sub rsp, 32

call malloc

add rsp, 32
ret
''')

class Function_header:
	def __init__(
		self, name, typeargs: tuple[str], args: tuple[str, str], ret_type: str,
		module, tell, line_no, infile=None, isextern=False
	):
		self.name = name
		self.typeargs = typeargs
		self.args = args
		self.ret_type = ret_type

		self.module = module
		self.tell = tell
		self.line_no = line_no
		self.infile = infile
		self.isextern = isextern

		self.instances = {}

		for arg_entry in args:
			if len(arg_entry) != 2:
				err(f'Invalid syntax {" ".join(arg_entry)!r}. '
					'Expected exactly 2 words for arg entry.')

	@staticmethod
	def parse(fn_qualname, arg_types, *, variables) -> tuple[Optional['Type'], 'Function_header']:
		global curr_mod

		if fn_qualname == 'alloc':
			return None, ALLOC_FN

		caller_type_name, dot, fn_name = fn_qualname.rpartition('.')

		if dot:
			parse_type_result = parse_type(caller_type_name, fn_types, variables=variables)
			if isinstance(parse_type_result, ParseTypeError):
				err(f'In {caller_type_name!r}, {parse_type_result}')
			if len(parse_type_result) != 1:
				err('Method calls expect exactly one type')
			caller_type, = parse_type_result
			# if caller_type.deref is not None:
			# 	caller_type = caller_type.deref

			# print('PARSED TYPE FOR METHOD:', caller_type)

			# NOTE: Temp
			if fn_name not in caller_type.methods:
				# print(f'TYPE {caller_type} HAS NO SUCH METHOD {fn_name!r}')
				# check for deref only if method doesn't exist
				if caller_type.deref is not None:
					caller_type = caller_type.deref
					if fn_name not in caller_type.methods:
						err(f'{caller_type} has no method named {fn_name!r}')
				else:
					err(f'{caller_type} has no method named {fn_name!r}')
			fn_header = caller_type.methods[fn_name]
			# print('METHOD  ', fn_name, fn_header)
		elif fn_name not in curr_mod.methods:
			err(f'No function named {fn_name!r}')
		else:
			fn_header = curr_mod.methods[fn_name]
			# print('FUNCTION', fn_name, fn_header)
			caller_type = None

		return caller_type, fn_header

	def __repr__(self):
		return (
			f'Function_header<{self.name!r} '
			f'in {self.infile and self.infile.name!r} '
			f'at line {self.line_no} (tell = {self.tell})>'
		)

	def add_sub(self, key: tuple[str], typeargs = ()) -> 'Function_instance':
		if len(key) != len(self.typeargs):
			err(f'Expected {len(self.typeargs)} type arguments '
				f'to {self.name!r}. Got {len(key)}')

		fn_instance = Function_instance(self, typeargs, len(self.instances))
		self.instances[key] = fn_instance
		return fn_instance

class Function_instance:
	def __init__(self, template, typeargs: list[str], id):
		self.template = template
		self.id = id
		self.type_mappings = dict(zip(template.typeargs, typeargs))
		self.arg_vars = {}
		self.offset = 0
		self.export_name = None

	def init_args_vars(self, types):
		local_types = types | self.type_mappings
		for typename, argname in self.template.args:
			if argname in self.arg_vars:
				err(f'Multiple instances of argument {argname!r} for function '
					f'{self.template.name!r}')

			parse_type_result = parse_type(typename, local_types, variables={})
			if isinstance(parse_type_result, ParseTypeError):
				err(f'In {typename!r}, {parse_type_result}')
			if len(parse_type_result) != 1:
				err('Type expression must evaluate to exactly one type')
			T, = parse_type_result

			self.offset += T.size
			self.arg_vars[argname] = Variable(
				argname, self.offset, T, self.template.line_no
			)

	def mangle(self):
		if self.template.isextern: return self.template.name.rpartition('.')[2]
		if self.export_name is not None: return self.export_name
		return f'_{self.id}{self.template.name}'

class Variable:  # Instantiated when types are known
	def __init__(
		self, name, offset, type, decl_line_no, *, field_id = None
	):
		self.name = name
		self.offset = offset
		self.type = type
		self.decl_line_no = decl_line_no
		self.field_id = field_id  # for enums

	def __repr__(self):
		return f'Variable({self.name!r})'

class Const:  # Instantiated when types are known
	def __init__(self, name, value, type, decl_line_no):
		self.name = name
		self.value = value
		self.type = type
		self.decl_line_no = decl_line_no
		self.fields = {}

	def __repr__(self):
		return f'Const({self.name} = {self.value!r})'

class Clause:
	def __init__(self, asm_str, *, size=8):
		if size not in (1, 2, 4, 8): err(f'Invalid clause size: {size}')

		self.asm_str = asm_str
		self.size = size

	@classmethod
	def ref(cls, addr, size=8):
		return cls(f'{size_prefix(size)} [{addr}]', size=size)

	def __repr__(self):
		return f'{self.__class__.__name__}({self.asm_str}, size={self.size})'

UNSPECIFIED_INT = type('UNSPECIFIED_INT', (), {
	'__repr__': lambda s: f'Type<UNSPECIFIED_INT>',
	'is_int': lambda s: True, 'deref': None,
})()
FLAG_TYPE = type('FLAG_TYPE', (), {
	'__repr__': lambda s: f'Type<FLAG>'
})()
MISSING = type('MISSING', (), {
	'__repr__': lambda s: f'<MISSING ARG>'
})()

class Type:
	def __init__(self, name, size = 0, args = (), is_enum = False):
		self.name = name
		self.size = size
		self.args = args

		self.deref = None
		self.ref = None
		self.parent = self

		self.children = {}   # nested types
		self.methods = {}

		self.last_field_id = None
		self.fields = {}
		self.consts = {}
		self.instances = {}  # polymorphic instances
		self.is_enum = is_enum

	def __repr__(self):
		return f'{self.__class__.__name__}({self.name})'

	@classmethod
	def get_size(cls, T, *, unspecified = 8):
		if T is UNSPECIFIED_INT:
			return unspecified
		if not isinstance(T, cls):
			err(f'[Internal Error] Invalid type {T.__class__} for type')
		return T.size


	@classmethod
	def read_module(cls, name, *, args=(), in_module=True):
		'''
		in_module=false means assume this as the root namespace... sort of
		'''

		global core_module, fn_queue, ALLOC_FN

		# First pass, get the declarations

		out_mod = cls(name, size=None, args=args)  # modules cannot be instantiated
		print(f'Start module {Shared.infile.name!r} {out_mod.name}')

		curr_mod_path = os.getcwd()

		if core_module is None:
			ALLOC_FN = Function_header('alloc', (), (('int', 'n'),), '', out_mod, 0, 0)
			ALLOC_FN.add_sub(())

			out_mod.methods |= {'alloc': ALLOC_FN}
		else:
			out_mod.methods = core_module.methods.copy()
			out_mod.consts = core_module.consts.copy()
			out_mod.children = builtin_types.copy()

			# print('Inherited core types:', out_mod.children)

		curr_type_dict = out_mod.children

		curr_type = out_mod
		type_stack = []

		in_function = False
		scope_level = 0

		tell = 0
		for Shared.line_no, Shared.line in enumerate(Shared.infile, 1):
			tell += len(Shared.line) + crlf
			match = Patterns.stmt.match(Shared.line)
			line = match[2]

			match = Patterns.split_word.match(line)


			if not match: continue
			elif match[1] == 'enum':
				# print('[P1] New type in', curr_type)

				if in_function: err('Local type definitions are not yet supported')

				name, *args = match[2].split()
				# if args: err('Polymorphic types are not yet supported')

				if name in curr_type_dict: err(f'Type {name!r} already defined')

				if in_module or curr_type is not out_mod:
					qual_name = f'{curr_type.name}.{name}'
				else:
					qual_name = name

				curr_type = Type(qual_name, args = curr_type.args + tuple(args), is_enum = True)

				type_stack.append(curr_type)
				curr_type_dict[name] = curr_type
				# print('NEW ENUM', curr_type)
				curr_type_dict = curr_type.children

				# scope_level += 1  # scope level goes up and down only if in_function

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

			elif match[1] == 'type':
				# print('[P1] New type in', curr_type)

				if in_function: err('Local type definitions are not yet supported')

				name, *args = match[2].split()
				# if args: err('Polymorphic types are not yet supported')

				if name in curr_type_dict: err(f'Type {name!r} already defined')

				if in_module or curr_type is not out_mod:
					qual_name = f'{curr_type.name}.{name}'
				else:
					qual_name = name
				
				curr_type = Type(qual_name, args = curr_type.args + tuple(args))

				type_stack.append(curr_type)
				curr_type_dict[name] = curr_type
				print('NEW TYPE', curr_type)
				curr_type_dict = curr_type.children

				# scope_level += 1  # scope level goes up and down only if in_function

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

			elif match[1] == 'export':
				renamed, qual_fn = match[2].split(maxsplit=1)
				print(f'{renamed = }, {qual_fn = }')

				fn_split = qual_fn.split(maxsplit=1)
				print(f'{fn_split = }')
				if len(fn_split) == 2:
					fn, type_str = fn_split
				else:
					fn, = fn_split
					type_str = ''

				if fn not in out_mod.methods:
					err(f'Function {fn!r} is not defined in {out_mod}. '
						'Cannot export.')

				args = parse_type(type_str, out_mod.children, variables=out_mod.consts)
				if isinstance(args, ParseTypeError):
					err(f'[in {type_str.lstrip()!r}] {args}')

				header = out_mod.methods[fn]
				if header.isextern:
					err(f'Cannot export extern function {header}')

				key = tuple(arg.name for arg in args)
				instance = header.add_sub(key)
				instance.export_name = renamed
				fn_queue.append((header, key))

				output(f'global {renamed}')

			elif match[1] == 'import':
				# print('[P1] Import')

				if in_function: err('Local type definitions are not yet supported')

				name, path_string = match[2].split(maxsplit=1)
				# if args: err('Polymorphic types are not yet supported')

				mod_path = parse_string(path_string)
				mod_path = os.path.expanduser(mod_path)
				mod_path = os.path.abspath(mod_path)

				if not os.path.exists(mod_path):
					err(f'Cannot import module. Path not found: {mod_path.decode()}')

				if name == 'extern':
					'''
					'import extern' adds a library file so that its functions
					can be introduced using 'extern'

					'extern as name' should be a thing. But, it's not.
					'''

					mod_dir, mod_name = os.path.split(mod_path.decode('utf-8'))

					if not (mod_name.startswith('lib') and mod_name.endswith('.a')):
						err('import extern requires file names of the form lib*.a')

					mod_name = mod_name.removeprefix('lib')
					mod_name = mod_name.removesuffix('.a')

					if mod_name not in Shared.library_set:
						Shared.library_set.add(mod_name)
						Shared.libraries.append(mod_name)  # link order matters
						Shared.library_paths.add(mod_dir)

					continue

				if name in curr_type_dict: err(f'Type {name!r} already defined')

				if in_module or curr_type is not out_mod:
					qual_name = f'{curr_type.name}.{name}'
				else:
					qual_name = name

				if mod_path in Shared.imports:
					# This makes it so that the same type imported multiple times
					# does not panic the type checker.
					# (Does not account for symbolic links)
					module = Shared.imports[mod_path]
				else:
					module_file = open(mod_path.decode('utf-8'))
					infile = Shared.infile
					Shared.infile = module_file
					os.chdir(os.path.dirname(mod_path))
					module = Type.read_module(
						qual_name, args = curr_type.args
					)
					os.chdir(curr_mod_path)
					Shared.infile = infile
					Shared.imports[mod_path] = module

				curr_type_dict[name] = module
				# print('NEW MODULE', module)

			elif match[1] == 'fn':  # function header
				# print(f'[P1] New function')
				if in_function:
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

				if name in curr_type.methods:
					err(f'Function {name!r} already defined.')

				if in_module or curr_type is not out_mod:
					qual_name = f'{curr_type.name}.{name}'
				else:
					qual_name = name

				fn = Function_header(
					qual_name, (*typeargs, *curr_type.args),
					tuple(arg.lstrip().rsplit(maxsplit=1) for arg in args),
					ret_type, out_mod, tell, Shared.line_no, Shared.infile
				)
				# print(f'FUNCTION {name}')
				curr_type.methods[name] = fn

				in_function = True
				scope_level += 1

			elif match[1] == 'extern':  # gets added to be linked
				# syntax extern name: type arg -> ret

				if in_function:
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

				if name in curr_type.methods:
					err(f'Function {name!r} already defined.')

				if in_module or curr_type is not out_mod:
					qual_name = f'{curr_type.name}.{name}'
				else:
					qual_name = name

				fn = Function_header(
					qual_name, (*typeargs, *curr_type.args),
					tuple(arg.lstrip().rsplit(maxsplit=1) for arg in args),
					ret_type, out_mod, None, Shared.line_no, Shared.infile,
					isextern = True
				)
				# print(f'EXTERN   {name}')
				curr_type.methods[name] = fn

				output('extern', name)

			elif not in_function:
				if match[1] == 'let':
					# print(f'[{Shared.line_no:3}] Detected statement type using', match and match[1])

					if not type_stack: err('Global variables are not yet supported')

					# print(f'[P1] New field in {curr_type!r}')
					field_id, alias = Patterns.alias_through_strings(match[2])
					match = Patterns.split_word.match(alias)
					name = match[1]
					T = match[2]

					if name in curr_type.fields:
						var = curr_type.fields[name]
						err(f'Field {name!r} already declared in '
							f'line {var.decl_line_no}')

					parse_type_result = parse_type(
						T, out_mod.children,
						variables=curr_type.fields
					)

					if not isinstance(parse_type_result, ParseTypeError):
						if len(parse_type_result) != 1:
							err('Field declaration expects exactly 1 type. '
								f'{T!r} resolves to '
								f'{len(parse_type_result)} types.')
						T, = parse_type_result
						# print('GOT REAL TYPE', T)
					else:
						print(f'{T.strip()!r} is not defined.')

					if curr_type.is_enum:
						offset = 0
					else:
						offset = curr_type.size

					if field_id is not None:
						if not curr_type.is_enum:
							err('Field alias is allowed only for enums types. '
								f'{curr_type} is a struct.')

						field_id = eval_const(field_id, types=types, variables=out_mod.consts)
						if not field_id.type.is_int():
							err('Enum field alias expected an integer. '
								f'Got {field_id.type}.')
						curr_type.last_field_id = field_id.value

					elif curr_type.last_field_id is None:
						curr_type.last_field_id = 0  # Default enum start id
					else:
						curr_type.last_field_id += 1

					curr_type.fields[name] = Variable(
						name, offset, T, Shared.line_no,
						field_id = curr_type.last_field_id,
					)
					# print(f'  Created a field {name!r} of {T}')
					if curr_type.size is not None:
						if not isinstance(T, Type):
							# print('curr_type.size:', f'Adding {T.size} to size')
							curr_type.size = None
						elif curr_type.is_enum:
							curr_type.size = max(curr_type.size, T.size)
						elif T.size is None:
							err(f'{T} is polymorphic. Cannot instantiate it without type arguments.')
						else:
							print(f'{curr_type}: adding {T} (size = {T.size})')
							curr_type.size += T.size
					# else:
						# print('curr_type.size:', 'Size is already None')

				elif match[1] == 'const':
					# print(f'[{Shared.line_no:3}] Detected statement type using', match and match[1])
					split = match[2].split(maxsplit=1)
					if len(split) != 2:
						err('An expression is required to declare a constant')

					name, exp = split

					if name in curr_type.consts:
						err(f'{name!r} is already a declared constant')

					# print(f'Declared a constant {name!r} associated to {curr_type}')
					const_var = eval_const(exp, curr_type_dict,
						variables=curr_type.consts)
					const_var.name = name
					curr_type.consts[name] = const_var
					# print(f'Constants in {curr_type}: {curr_type.consts}')

				elif match[1] == 'end':
					# print(f'[P1] End of {curr_type!r}, size = {curr_type.size}')
					# for name, field in curr_type.fields.items():
						# print('   ', name, field.type)
					# print()
					type_stack.pop()  # no need to check emptiness.

					if curr_type.is_enum:
						if curr_type.size is None:
							err('Polymorphic enums are not yet supported.')

						print('Getting discr size of', curr_type)
						if curr_type.last_field_id is not None:
							curr_type.last_field_id = max(
								field.field_id
								for field in curr_type.fields.values()
							)
							discriminator_size = get_discriminator_size(
								curr_type.last_field_id
							)

							curr_type.size += discriminator_size
							for field in curr_type.fields.values():
								field.offset += discriminator_size

						print('END ENUM', curr_type)
					else:
						print('END TYPE', curr_type)

					print(f'{curr_type} has a size of {curr_type.size}')

					if type_stack: curr_type = type_stack[-1]
					else: curr_type = out_mod

					curr_type_dict = curr_type.children

			# else: not (type_stack and not in_function) == not type_stack or in_function

			elif match[1] in ('if', 'while'):
				# print(f'[P1] Enter construct')

				scope_level += 1

			elif match[1] == 'end':
				# first part handled by match[1] == 'type'
				# (type > ... > type) > fn > cond > cond > ...
				scope_level -= 1
				if scope_level < 0:
					err('end statement does not match any block')
				elif scope_level == 0:
					in_function = False
				# else:
				# 	print(f'[P1] Exit construct')

		print('End module', out_mod.name)

		return out_mod


	def get_instance(self, args: tuple['Type']) -> 'Type':
		global types

		# if self is PTR_TYPE:
			# print('GETTING PTR INSTANCE:', args)

		if self.parent is not self:
			if args:
				err('Cannot instantiate with arguments if already instantiated')
			return self

		if len(args) != len(self.args):
			err(f'{self} expects {len(self.args)} polymorphic arguments. '
				f'Got {len(args)}: {args}.')

		if args in self.instances:
			return self.instances[args]

		# print(f'INSTANTIATE NEW VARIANT OF {self.name}{args}, enum: {self.is_enum}')

		if not args:
			# not polymorphic

			self.instances[args] = self
			return self
			# if not self.fields:
			# 	# print('Instantiated a type that has no fields, size =', self.size, [self])
			# 	return self  # for fn_wrapper types?
			# instance = self
			# return instance
		else:
			local_types = dict(zip(self.args, args))
			instance_types = types | local_types

			instance = Type(
				' '.join(T.name for T in [self, *args]), is_enum=self.is_enum
			)
			instance.args = args
			instance.parent = self
			instance.methods = self.methods
			self.instances[args] = instance

		# print('GET INSTANCE FIELDS OF', self, self.fields, f'({instance.fields = })')

		if self.is_enum:
			# (max_field_id).bit_length() is the number of bits required
			# (x-1)//n+1 rounds up?
			discriminator_size = get_discriminator_size(self.last_field_id)

		for name, field in self.fields.items():
			T = field.type
			if isinstance(T, str):
				parse_type_result = parse_type(T, instance_types, variables={})

				if isinstance(parse_type_result, ParseTypeError):
					err(f'Error parsing {T!r}: {parse_type_result}')
				if len(parse_type_result) != 1:
					err('Declaration requires exactly one type. '
						f'Got {len(parse_type_result)}')

				T, = parse_type_result

			if self.is_enum:
				offset = discriminator_size
			else:
				offset = instance.size
				instance.size += T.size

			# print(f'{instance.fields = }')

			field = Variable(
				name, offset, T, field.decl_line_no,
				field_id = len(instance.fields),
			)
			# print(f'Creating {field} with offset {field.offset} and id {field.field_id}')
			instance.fields[name] = field

		if self.is_enum:
			instance.size = (
				discriminator_size
				+ max((var.type.size for var in instance.fields.values()), default=0)
			)

		return instance

	def match_pattern(self, type_str, module) -> dict['type_arg': 'Type']:
		type_queue = [self]
		out_mappings = {}
		types = module.children
		# print('Matching Pattern', self, type_str)

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
				# print(f'deref: {token!r} -> {token[1:]!r}; {expected_type} -> {expected_type.deref}')
				expected_type = expected_type.deref
				token = token[1:]
			elif expected_type.deref is not None:
				if token in types:
					# print(f'  {types[token] = }, {expected_type = }')
					if types[token].deref == expected_type.deref:
						expected_type = types[token]

			# What to do for T.child_type if T is polymorphic?
			# it can be the same type, but have different parents
			# eg multiple files import the same module
			qual_token = token
			token, *children = qual_token.split('.')

			if token in types: # token is a concrete type

				evaluated_token = types[token]
				for child in children:
					if child not in evaluated_token.children:
						err(f'{child!r} is not defined in {evaluated_token!r}')

					evaluated_token = evaluated_token.children[child]

				# expected_type.parent is the polymorphic type
				if evaluated_token not in (ANY_TYPE, expected_type.parent):
					err(f'{qual_token!r} evaluated to {evaluated_token}, '
						f'but the argument expected {expected_type}.')

				type_queue.extend(expected_type.args)

			# token is a polymorphic type
			elif children:
				# TODO: suppress error if a mapping is already made
				err('Children of polymorphic types are not yet supported')

			elif token not in out_mappings:
				# print(f'MATCH {token!r} to {expected_type}')
				out_mappings[token] = expected_type
			elif out_mappings[token] is UNSPECIFIED_INT:
				# print(f'MATCH {token!r} to {expected_type}')
				if not expected_type.is_int():
					err(f'Expected an integer type for {token!r}')
				out_mappings[token] = expected_type
			elif expected_type is ANY_TYPE:
				# print(f'MATCH {token!r} to {expected_type}')
				out_mappings[token] = expected_type

			elif out_mappings[token] is not expected_type:
				err('Multiple mappings to same argument. '
					f'Trying to map {token!r} to '
					f'{expected_type} and {out_mappings[token]}')

		return out_mappings

	def pointer(self):
		if self.ref is not None: return self.ref

		self.ref = PTR_TYPE.get_instance((self,))
		self.ref.name = '&' + self.name
		self.ref.deref = self
		self.ref.size = PTR_TYPE.size

		# print('POINTER created:', self.ref)
		# print('POINTER size:   ', self.ref.size)
		# print('POINTER methods:', self.ref.methods)

		# print(self.ref, 'has a deref of', self)
		return self.ref

	def is_int(self):
		return self in (INT_TYPE, CHAR_TYPE, U64_TYPE)

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
	ctrl_no = 0

	@classmethod
	def next(cls):
		out = cls.ctrl_no
		cls.ctrl_no += 1
		return out

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
	ESC  = auto()
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

def parse_type(type_str, types, *, variables) -> Union[list[Type], ParseTypeError]:
	if type_str.startswith('&'):
		type_str = type_str[1:].strip()
		pointer = True
	else:
		pointer = False

	type_tokens = type_str.split(maxsplit=1)

	if not type_tokens:
		return []

	root_token = type_tokens[0]

	# print(f'PARSE TYPE {type_str!r}.split() = {type_tokens}')

	token, _, rhs = root_token.rpartition(':')
	field, *children = rhs.split('.')

	if token:
		# print(f'Type meta:  {token = }, {field = }')


		if field == 'type':
			_insts, _clauses, exp_type = parse_token(token, types,
				variables=variables, virtual=True)
		else:
			err('Unsupported metadata field. '
				f"':{field}' does not yield a type.")
		T = exp_type
		if T is UNSPECIFIED_INT:
			T = INT_TYPE
		elif isinstance(T, Flag):
			err('Using booleans is supported only in conditionals')

	else:
		T = field
		if T not in types:
			# print(repr(T), 'not in', types)
			return ParseTypeError(f'Type {T!r} is not defined')
		else:
			T = types[T]

	for child in children:
		# print(f'Getting child {T!r}.{child}')
		if child not in T.children:
			err(f'Type {child!r} is not defined in {T!r}')

		T = T.children[child]

	# print(f'PARSE CONVERT {root_token!r} -> {T}')

	if len(type_tokens) > 1:
		args = parse_type(type_tokens[1], types, variables=variables)
		if isinstance(args, ParseTypeError): return args
		# print(f'PARSETYPE RECURSION with {type_tokens[1]!r} yielded {args}')
	else:
		args = []

	if T.parent is not T: args_len = 0
	else: args_len = len(T.args)

	# print('PARSETYPE getting instance with', args_len, 'args:', args)

	instance = T.get_instance(tuple(args[:args_len]))
	if pointer: instance = instance.pointer()

	args[:args_len] = instance,
	return args

def parse_string(token) -> bytes:
	if token[0] != '"': err('Strings can only start with \'"\'')

	string_data = bytearray()
	h_val = None
	escape = In_string.NONE
	for c in token[1:]:
		if escape is In_string.NONE:
			if   c == '\\': escape = In_string.ESC
			elif c == '\"': escape = In_string.OUT
			else: string_data.extend(c.encode())
		elif escape is In_string.ESC:
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
			h_val = int(c, 16) << 4
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

	return bytes(string_data)

def parse_token(token: 'stripped', types, *, variables, virtual=False) \
	-> (list[str], tuple[Optional[int], ...], Type):
	'''
	Returns a set of instructions and the clauses to access the result.
	The clauses mention how to access the different part of the result.
	'''

	# (instructions to get the value of token, expression, type)

	# print('Parse token', repr(token))

	idx = Patterns.find_through_strings(token, '{')
	if idx != -1:
		if token[-1] != '}':
			err("Enum token must end with '}'")

		enum_val_token = token[idx+1:-1].strip()

		if not enum_val_token:
			insts = []
			clauses = ()
			T = VOID_TYPE
		else:
			insts, clauses, T = parse_token(
				enum_val_token, types, variables=variables, virtual=virtual
			)

		type_str, _, variant_name = token[:idx].rpartition('.')

		parse_type_result = parse_type(type_str, types, variables=variables)

		if isinstance(parse_type_result, ParseTypeError):
			err(f'In {type_str!r}, {parse_type_result}')
		if len(parse_type_result) != 1:
			err(
				f'Expected exactly 1 type for enum. '
				f'Got {len(parse_type_result)}'
			)

		Enum_type, = parse_type_result

		if not Enum_type.is_enum: err(f'{Enum_type} is not an enum.')

		variant_name = variant_name.strip()

		if variant_name not in Enum_type.fields:
			err(f'{variant_name!r} is not a variant of {Enum_type}')

		variant = Enum_type.fields[variant_name]

		# type checking
		if T is not variant.type:
			if not (T is UNSPECIFIED_INT and variant.type.is_int()):
				err(
					f'Variant {variant_name!r} of {Enum_type} expects '
					f'{variant.type}. Got {T} instead.'
				)

		discriminator_size = get_discriminator_size(Enum_type.last_field_id)
		if discriminator_size:
			clauses = (
				Clause(
					f'{variant.field_id}',
					size=get_discriminator_size(Enum_type.last_field_id)
				),
				*clauses,
			)

		output('; Enum clauses:', clauses)

		return insts, clauses, Enum_type

	for operator in ('<=', '>=', '<<', '>>', '==', '!=', '->', *'<>|^&+-*/%'):  # big ones first
		operator_idx = Patterns.find_through_strings(token, operator)
		if operator_idx != -1:
			l_operand = token[:operator_idx].strip()
			r_operand = token[operator_idx+len(operator):].strip()

			# print(f'{l_operand!r} {operator} {r_operand!r}')
			if not l_operand:
				# err('[Internal error] Unary not checked earlier')
				r_operand = None
				continue

			o_insts, o_clauses, o_type = parse_token(r_operand, types,
				variables=variables)

			# if len(o_clauses) != 1:
			# 	err('Operations are not supported for non-standard sizes. '
			# 		f'{o_type} has a size of {o_type.size} bytes. '
			# 		f'({len(o_clauses)} clauses)')
			if o_insts: err(f'Expression {token!r} is too complex')  # we do this to prevent overwriting registers
			token = l_operand
			break
	else:
		operator = None

	if token.startswith('&'):
		addr = Address_modes.ADDRESS
		token = token[1:].lstrip()
	elif token.startswith('*'):
		addr = Address_modes.DEREF
		token = token[1:].lstrip()
	else:
		addr = Address_modes.NONE

	clauses = ()
	val = None
	var = None
	T   = None
	insts = []

	colon_idx = Patterns.rfind_through_strings(token, ':')
	dot_idx = Patterns.find_through_strings(token, '.')

	if colon_idx != -1:
		exp = token[:colon_idx]
		field = token[colon_idx+1:].strip()

		# TODO: T, val = parse_meta(exp, field)

		# print(f'Token meta: {exp = }, {field = }')

		if exp.endswith(':'):  # const
			parse_type_result = parse_type(exp[:-1], types, variables=variables)
			if isinstance(parse_type_result, ParseTypeError):
				err(f'In {exp!r}, {parse_type_result}')
			T, = parse_type_result

			if field not in T.consts:
				err(f'{field!r} was not found in {T}')

			val = T.consts[field]

			T = val.type
			val = val.value

		elif field == 'size':
			parse_type_result = parse_type(exp, types, variables=variables)
			if isinstance(parse_type_result, ParseTypeError):
				err(f'In {exp!r}, {parse_type_result}')
			if addr is Address_modes.ADDRESS:
				val = PTR_SIZE
			else:
				if len(parse_type_result) != 1:
					err(f'{exp!r} does not correspond to a single type')
				T, = parse_type_result
				val = T.size
			# print(f'{T}:size = {val}')
			T = UNSPECIFIED_INT
		elif field == 'type':
			_insts, _clauses, exp_type = parse_token(exp, types,
				variables=variables, virtual=True)

			if addr is Address_modes.DEREF:
				addr = Address_modes.NONE

				if exp_type.deref is None:
					err(f'{exp_type} cannot be dereferenced')
				exp_type = exp_type.deref

			string = bytes(exp_type.name, 'utf-8')
			if virtual:
				clauses = (Clause('_dummy_string_label'),)
			else:
				clauses = (Clause(get_string_label(string, strings)),)
			T = STR_TYPE
		elif field == 'disc':
			f_insts, clauses, exp_type = parse_token(exp, types,
				variables=variables)
			insts += f_insts

			if addr is Address_modes.DEREF:
				addr = Address_modes.NONE

				if exp_type.deref is None:
					err(f'{exp_type} cannot be dereferenced')
				exp_type = exp_type.deref

			if exp_type.deref is not None:
				err('Implicit dereferencing not allowed for enum variant id')

			if not exp_type.is_enum:
				err(f"Meta field 'disc' expected an enum. {exp_type} is not an enum.")

			size = get_discriminator_size(exp_type.last_field_id)
			clause = clauses[0]

			# TODO: `clause.deref` instead of `clauses`
			if clause.size != size:
				err(f'Cannot get discriminator for enum {T} with size {T.size}')

			clauses = (clause,)

			T = UNSPECIFIED_INT

		elif field == 'name':
			parse_type_result = parse_type(exp, types, variables=variables)
			if isinstance(parse_type_result, ParseTypeError):
				err(f'In {exp!r}, {parse_type_result}')
			if len(parse_type_result) != 1:
				err(f'{exp!r} does not correspond to a single type')
			T, = parse_type_result

			string = bytes(T.name, 'utf-8')
			clauses = (Clause(get_string_label(string, strings)),)
			T = STR_TYPE

		elif field == 'null':
			parse_type_result = parse_type(exp, types, variables=variables)
			if isinstance(parse_type_result, ParseTypeError):
				err(f'In {exp!r}, {parse_type_result}')
			if len(parse_type_result) != 1:
				err(f'{exp!r} does not correspond to a single type')
			T, = parse_type_result

			clauses = (Clause('0'),)
			T = T.pointer()

		else:  # TODO: var:len
			err(f'Unsupported metadata field {field!r} for token')

		if addr is Address_modes.ADDRESS:
			err('Taking address of a meta field; not allowed')
		elif addr is Address_modes.DEREF:
			err('Dereferencing a meta field; not allowed')

	elif token.isidentifier():
		# if addr: err("Can't take addresses of local variables yet")
		# print(f'{variables = }')
		if token not in variables: err(f'{token!r} not defined')
		var = variables[token]

		if isinstance(var, Const):
			if addr is Address_modes.ADDRESS:
				err('Cannot take a reference to a constant')

			T = var.type
			val = var.value  # TODO: enums

		else:
			offset = var.offset
			T = var.type
			if addr is Address_modes.ADDRESS:
				T = T.pointer()
				insts.append(
					f'lea {{dest_reg:{Type.get_size(T)}}}, [rsp + {offset}]'
				)
				clauses = ()
			elif virtual:
				clauses = ()
			else:
				# clause = f'{size_prefix(var.type.size)} [{clause}]'

				# big moves
				sizes = split_size(T.size)
				clauses = []
				for size in sizes:
					clauses.append(Clause(
						f'{size_prefix(size)} [rsp + {offset}]', size=size
					))
					offset += size

	elif dot_idx != -1:
		root = token[:dot_idx]
		chain = token[dot_idx+1:].split('.')
		root = root.strip()
		if root not in variables:
			err(f'{root!r} not defined')

		# print(f'Getting a field of {root!r}')
		var = variables[root]
		offset = var.offset
		base_reg = 'rsp'

		T = var.type
		# print(f'Getting a field of {root!r} {T}')
		for field in chain:

			field = field.strip()
			# print(f'  {field = }')
			if isinstance(T, Flag):
				err('Accessing fields of an enum variant check '
					'is not supported.')

			if T is not STR_TYPE and T.deref is not None:
				# We want to dereference T, so we first put it into a register
				size = Type.get_size(T)  # always equal to PTR_SIZE
				insts.append(
					f'mov {{dest_reg:{size}}}, '
					f'{size_prefix(size)} [{base_reg} + {offset}]'
				)
				base_reg = f'{{dest_reg:{size}}}'
				offset = 0
				T = T.deref

			if field not in T.fields: err(f'{T} has no field {field!r}')

			var = T.fields[field]

			# print(f'  {T = } {T.is_enum = }')
			if T.is_enum:
				discriminator_size = get_discriminator_size(T.last_field_id)

				insts.append(
					# crashes on non-standard field_sizes
					f'cmp {size_prefix(discriminator_size)} '
					f'[{base_reg} + {offset}], {var.field_id}'
				)

				T = Flag.e
			else:
				# for _name, _field in T.fields.items():
					# print(' ', _name, _field.type)
				# print(f'  {T}.fields[{field!r}]')
				T = var.type

			offset += var.offset
			# print(f'  offset: {offset} ({var.offset = })')

		base_addr = f'{base_reg} + {offset}'
		if addr is Address_modes.ADDRESS:
			T = T.pointer()
			insts.append(f'lea {{dest_reg:{Type.get_size(T)}}}, [{base_addr}]')
			clauses = ()
		elif isinstance(T, Flag):
			clauses = ()
		else:
			# DONE: big move support
			# TODO: it needs to support stacks eventually
			split_size(T.size)
			clauses = []

			for size in split_size(Type.get_size(T)):
				clauses.append(Clause(
					f'{size_prefix(size)} [{base_reg} + {offset}]', size=size
				))
				offset += size

	elif token.isdigit():
		if addr is Address_modes.ADDRESS:
			err("Can't take address of an integer literal")
		if addr is Address_modes.DEREF:
			err("Can't dereference an integer literal")
		val = int(token)
		T = UNSPECIFIED_INT

	elif token.startswith("'"):
		if addr is Address_modes.ADDRESS:
			err("Can't take address of a character literal")
		if addr is Address_modes.DEREF:
			err("Can't dereference a character literal")

		if len(token) == 1:
			err('"\'" is not a valid character literal.')

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
		T = CHAR_TYPE

	elif token.startswith('"'):
		if addr is Address_modes.ADDRESS: err('Cannot take address of string literal')

		string = parse_string(token)
		clauses = (Clause(get_string_label(string, strings)),)
		T = STR_TYPE

	else:
		err(f'Invalid token syntax {token!r}')

	if T is None:
		err(f'[INTERNAL ERROR] Type of token {token!r} not determined.')

	if val is not None:
		clauses = (Clause(f'{val}', size = Type.get_size(T)),)

	if addr is Address_modes.DEREF:
		if T.deref in (None, ANY_TYPE):
			err(f'Cannot dereference a value of type {T}')
		if len(clauses) != 1: err('Invalid pointer size')

		clause, = clauses
		size = Type.get_size(T.deref)
		split_sizes = split_size(size)

		ptr_reg = dest_reg_fmts[len(split_sizes)-1]

		insts.append(
			f'mov {{{ptr_reg}:{clause.size}}}, {clause.asm_str}'
		)

		offset = 0
		clauses = []
		output(';', dest_reg_fmts, size, split_size(size))
		for dest_reg_fmt, sub_size in zip(dest_reg_fmts, split_size(size)):
			clauses.append(Clause.ref(
				f'{{{ptr_reg}:{Type.get_size(T)}}} + {offset}'
			))
			offset += sub_size
		output(';', clauses)
		T = T.deref

	if operator == '->':  # this is not your typical operator
		if var is None or not isinstance(T, Flag):
			err("Expected an enum variant before '->'")

		if var.type is not o_type:
			err(
				f'Enum unwrap returns {var.type}. '
				f'Trying to assign to {o_type} instead'
			)

		size = var.type.size

		ctrl_no = Ctrl.next()

		print(f'{Shared.line_no}: ARROW FROM {clauses} TO {o_clauses} ({size = })')

		insts += [
			# preserves flag state

			f'j{(~T).name} _U{ctrl_no}',
		]
		clause_offset = offset
		for clause_size, o_clause in zip(split_size(size), o_clauses):
			insts += [
				f'mov {{{dest_reg_fmts[0]}:{clause_size}}}, '
				f'{size_prefix(clause_size)} [{base_reg} + {clause_offset}]',
				f'mov {o_clause.asm_str}, {{{dest_reg_fmts[0]}:{clause_size}}}',
			]
			clause_offset += clause_size
		insts += [
			f'_U{ctrl_no}:',
		]

	elif operator is not None:
		# Type check before codegen
		if len(o_clauses) != 1:
			err(f'Operators are not supported for non-standard sizes. '
				f'({T} uses {T.size} bytes)')

		T = operator_result_type(operator, T, o_type)
		o_clause, = o_clauses

		insts += [
			# *o_insts,  # too complex if not empty

			*(f'mov {{{dest_reg_fmt}:{clause.size}}}, {clause.asm_str}'
				for dest_reg_fmt, clause in zip(dest_reg_fmts, clauses)),

			*get_operator_insts(operator, o_clause, o_type)
		]
		clauses = ()
		# print(f'OPERATOR {operator!r} using {T} and {o_type} ({addr = }) gives... ', end='')
		# print(T)
	return insts, clauses, T

def eval_const(exp, types, *, variables) -> Const:
	# don't support enum{} yet

	for operator in ('<=', '>=', '<<', '>>', '==', '!=', *'<>|^&+-*/%'):  # big ones first
		operator_idx = Patterns.find_through_strings(exp, operator)
		if operator_idx != -1:
			err(f'Operators in constants are not yet supported')

			l_operand = exp[:operator_idx].strip()
			r_operand = exp[operator_idx+len(operator):].strip()

			# print(f'  {l_operand!r} {operator} {r_operand!r}')
			if not l_operand:
				r_operand = None
				continue

			o_val = eval_const(r_operand, types,
				variables=variables)
			exp = l_operand
			break

	if exp.startswith('&'):
		addr = Address_modes.ADDRESS
		exp = exp[1:].lstrip()
	elif exp.startswith('*'):
		err("Can't save a dereferenced value as a constant")
	else:
		addr = Address_modes.NONE

	clauses = ()
	val = None
	r_operand = None
	T = None

	colon_idx = Patterns.rfind_through_strings(exp, ':')
	dot_idx = Patterns.find_through_strings(exp, '.')

	if colon_idx != -1:
		exp = exp[:colon_idx]
		field = exp[colon_idx+1:].strip()

		# TODO: T, val = parse_meta(exp, field)

		# print(f'Token meta: {exp = }, {field = }')

		if exp.endswith(':'):  # const defined in terms of a const
			parse_type_result = parse_type(exp[:-1], types, variables=variables)
			if isinstance(parse_type_result, ParseTypeError):
				err(f'In {exp!r}, {parse_type_result}')
			T, = parse_type_result

			if field not in T.consts:
				err(f'{field!r} was not found in {T}')

			val = T.consts[field]

			T = val.type
			val = val.value

		elif field == 'size':
			parse_type_result = parse_type(exp, types, variables=variables)
			if isinstance(parse_type_result, ParseTypeError):
				err(f'In {exp!r}, {parse_type_result}')
			if addr is Address_modes.ADDRESS:
				val = PTR_SIZE
			else:
				if len(parse_type_result) != 1:
					err(f'{exp!r} does not correspond to a single type')
				T, = parse_type_result
				val = T.size
			# print(f'{T}:size = {val}')
			T = UNSPECIFIED_INT
		elif field == 'type':
			_insts, _clauses, exp_type = parse_token(exp, types,
				variables=variables, virtual=True)

			if addr is Address_modes.DEREF:
				addr = Address_modes.NONE

				if exp_type.deref is None:
					err(f'{exp_type} cannot be dereferenced')
				exp_type = exp_type.deref

			string = bytes(exp_type.name, 'utf-8')
			if virtual:
				val = None
			else:
				val = get_string_label(string, strings)
			T = STR_TYPE

		elif field == 'disc':
			err('Discriminator of an expression is not a constant')

		elif field == 'name':
			parse_type_result = parse_type(exp, types, variables=variables)
			if isinstance(parse_type_result, ParseTypeError):
				err(f'In {exp!r}, {parse_type_result}')
			if len(parse_type_result) != 1:
				err(f'{exp!r} does not correspond to a single type')
			T, = parse_type_result

			string = bytes(T.name, 'utf-8')
			clause = get_string_label(string, strings)
			T = STR_TYPE

		elif field == 'null':
			parse_type_result = parse_type(exp, types, variables=variables)
			if isinstance(parse_type_result, ParseTypeError):
				err(f'In {exp!r}, {parse_type_result}')
			if len(parse_type_result) != 1:
				err(f'{exp!r} does not correspond to a single type')
			T, = parse_type_result

			clause = '0'
			T = T.pointer()

		else:  # TODO: var:len
			err(f'Unsupported metadata field {field!r} for token')

	elif exp.isidentifier():
		# if addr: err("Can't take addresses of local variables yet")
		if exp not in variables: err(f'{exp!r} not defined')
		var = variables[exp]

		if addr is Address_modes.ADDRESS:
			err('Reference constants are not yet supported')
		if not isinstance(var, Const):
			err('Cannot define a constant using a variable')

		T = var.type
		val = var.value

	elif dot_idx != -1:
		err('Constant fields are not yet supported')

	elif exp.isdigit():
		if addr is Address_modes.ADDRESS:
			err("Can't take address of an integer literal")
		if addr is Address_modes.DEREF:
			err("Can't dereference an integer literal")
		val = int(exp)
		T = UNSPECIFIED_INT

	elif exp.startswith("'"):
		if addr is Address_modes.ADDRESS:
			err("Can't take address of a character literal")
		if addr is Address_modes.DEREF:
			err("Can't dereference a character literal")
		if exp[-1] != "'":
			err('Expected end quote (\') at the end of character literal.')

		if exp[1] != '\\':
			if len(exp) != 3:
				err('Invalid syntax for character literal')
			val = ord(exp[1])
		elif exp[2] != 'x':
			if len(exp) > 4: err('Character literal too long')
			if len(exp) < 4: err('Character literal too short')
			c = exp[2]
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
			if len(exp) != 6:
				err('Invalid syntax for character literal')
			val = int(exp[3:5], 16)
		T = CHAR_TYPE

	elif exp.startswith('"'):
		if addr is Address_modes.ADDRESS: err('Cannot take address of string literal')

		string = parse_string(exp)
		val = get_string_label(string, strings)
		T = STR_TYPE

	else:
		err(f'Invalid token syntax {exp!r}')

	if T is None:
		err(f'[INTERNAL ERROR] Type of token {exp!r} not determined.')

	if val is None:
		err('[Internal Error] Got no val while evaluating a constant')

	if addr is Address_modes.DEREF:
		err('Dereferencing does not yield a constant')

	l_operand = Const('_l', val, T, Shared.line_no)

	if r_operand is not None:
		err(f'Operators in constants are not yet supported')
		# TODO: define constant operator
		# T, val = eval_operator_const(l_operand, operator, r_operand)
		# return Const('_val', val, T, Shared.line_no)

	return l_operand


# muddles rbx if dest_reg is Flag
def parse_exp(exp: 'stripped', *, dest_regs, fn_queue, variables) -> Type:
	# TODO: return const when regs not required

	# extract call_function(fn, args)
	# print(f'PARSE EXP: {exp!r}')

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

		insts, exp_clauses, T = parse_token(exp, fn_types, variables=variables)
		# print('Token', repr(exp), 'has a type of', T)

		if isinstance(T, Flag):
			# print('Parsed flag token', exp, '->', T)
			for inst in insts:
				output(inst.format(dest_reg=Register.b))
				# err('[Internal error] Multiple instructions from a flag token')

			if dest_regs is Flag: return T  # returns flag if relational

			if T.value is Flag.ALWAYS:  exp_clauses = (Clause('1'),)
			elif T.value is Flag.NEVER: exp_clauses = (Clause('0'),)

			# [x] T flag, dest_reg val -> setcc; ALWAYS/NEVER converted to values
			else:
				output(f'set{T.name} {dest_regs[0].encode(size=8)}')
				for dest_reg in dest_regs[1:]:
					output(
						f'xor {dest_reg.encode(size=8)}, '
						f'{dest_reg.encode(size=8)}'
					)

				return UNSPECIFIED_INT

			T = UNSPECIFIED_INT

		size = Type.get_size(T)

		# [x] T val, dest_reg val -> mov
		if dest_regs is not Flag:
			# This is important. This is responsible for assignment.
			if len(dest_regs) > 2 or len(dest_regs) == 0:
				err(
					'[Internal error] Unsupported variable size: '
					f'{Type.get_size(T)}'
				)
			elif len(dest_regs) == 2:
				dest_reg = dest_regs[0]
				aux_reg  = dest_regs[1]
			elif len(dest_regs) == 1:
				dest_reg = dest_regs[0]
				aux_reg = None

			for inst in insts:
				output(inst.format(dest_reg=dest_reg, aux_reg=aux_reg))

			output(';', dest_regs, exp_clauses, size)
			for dest_reg, sub_clause in zip(dest_regs, exp_clauses):
				# print('Moving into rax', T, size)
				output(
					f'mov {dest_reg:{sub_clause.size}}, '
					f'''{sub_clause.asm_str.format(
						dest_reg=dest_reg, aux_reg=aux_reg
					)}'''
				)

			return T
		else:
			# [ ] T val, dest_reg flag -> test

			if len(exp_clauses) > 1:
				# TODO: if arr
				err('Non-standard sized expressions not allowed as a condition')

			if len(exp_clauses) == 1:
				exp_clause, = exp_clauses
			else:
				exp_clause = None

			for inst in insts:
				output(
					inst.format(
						dest_reg=standard_dest_regs[0],
						aux_reg=standard_dest_regs[1],
					)
				)

			if T is UNSPECIFIED_INT and exp_clause.asm_str.isdigit():
				val = int(exp_clause.asm_str)
				if val: return Flag.ALWAYS
				else: return Flag.NEVER

			elif T is STR_TYPE:
				# TODO: account for null strings
				return Flag.ALWAYS
			elif exp_clause is not None:
				# works only if exp_clause can be a dest
				output(f'''test {exp_clause.asm_str.format(
						dest_reg=standard_dest_regs[0],
						aux_reg=standard_dest_regs[1],
					)}, -1''')
				return Flag.nz
			else:
				output(f'test {{0:{T.size}}}, {{0:{T.size}}}'
					.format(Register.a))
				return Flag.nz

	# print('Parse exp', repr(exp))

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

		# First argument
		insts, clauses, T = parse_token(exp[:idx].strip(), fn_types, variables=variables)
		# print(f'Type of {exp[:idx]!r} is {T}')

		# TODO: What if I want an integer of a different type?

		if len(clauses) != 1:
			err('Non standard sizes are not yet supported in function calls')

		for inst in insts:
			output(inst.format(dest_reg=arg_regs[0], aux_reg=arg_regs[1]))
		for clause, arg_reg in zip(clauses, arg_regs):
			if clause is None: continue
			reg_str = arg_regs[0].encode(size=clause.size)
			output(
				f'mov {reg_str}, '
				f'''{clause.asm_str.format(
					dest_reg=arg_regs[0], aux_reg=arg_regs[1]
				)}'''
			)

		arg_types = [T]
		if fn_name not in T.methods:
			if T.deref is None or fn_name not in T.deref.methods:
				err(f'{T} has no method {fn_name!r}')
			T = T.deref

		fn_header = T.methods[fn_name]
		ret_type = call_function(
			fn_header, arg_types, exp[idx:],
			variables=variables, caller_type=T,
		)

	else:
		fn_name = exp[:idx].strip()
		if fn_name.startswith('*'):
			fn_deref = True
			fn_name = fn_name[1:].strip()
		else:
			fn_deref = False
		arg_types = []

		T, fn_header = Function_header.parse(fn_name, arg_types, variables=variables)
		ret_type = call_function(
			fn_header, arg_types, exp[idx:],
			variables=variables, caller_type=T,
		)

		if fn_deref:
			if ret_type.deref is None:
				err(f'Could not dereference {ret_type}')
			ret_type = ret_type.deref
			output(f'mov {Register.a:{ret_type.size}}, {size_prefix(ret_type.size)} [rax]')

	# print(f'{dest_reg = }')
	if dest_regs is Flag:
		# output is in rax
		# print('Classified as a flag')
		output(f'test {Register.a:{ret_type.size}}, '
			f'{Register.a:{ret_type.size}}')
		return Flag.nz

	# print('Not classified as a flag')

	return ret_type


# args_str must include the starting bracket
# args_str = '(arg1, arg2)', but not 'arg, arg2)'
def call_function(fn_header, arg_types, args_str, *, variables, caller_type = None):
	if fn_header is ALLOC_FN:
		alloc_type = None
		alloc_fac  = None

	idx = 0
	while idx != -1:
		if len(arg_types) >= len(arg_regs):
			err(f'Only upto {len(arg_regs)} arguments are allowed')
		arg_reg = arg_regs[len(arg_types)]

		end = Patterns.find_through_strings(args_str, ',', start=idx+1)
		arg = args_str[idx+1:end].strip()

		idx = end
		if not arg and idx == -1: break

		if fn_header is ALLOC_FN:
			# doesn't work if only one argument is provided. Should work.
			if alloc_type is None:
				parse_type_result = parse_type(arg, fn_types, variables=variables)

				if isinstance(parse_type_result, ParseTypeError):
					err(f'In {arg!r}, {parse_type_result}')
				if len(parse_type_result) != 1:
					err('Expected exactly one type in alloc()')

				alloc_type, = parse_type_result
				# if alloc_type is ANY_TYPE:
				# 	err(f'{alloc_type} has no associated size')
				alloc_fac = alloc_type.size
				continue

			arg = f'{alloc_fac}*{arg}'

		# print('Arg:', arg)

		insts, clauses, T = parse_token(arg, fn_types, variables=variables)
		# print(f'Type of {arg!r} is {T}')

		# TODO: What if I want an integer of a different type?

		for inst in insts:
			output(inst.format(dest_reg=arg_reg))

		if len(clauses) > 1:
			err(
				'[Internal Error] Only register sized arguments are supported. '
				f'({arg!r} has a size of {Type.get_size(T)})'
			)
		elif clauses:
			clause, = clauses
			reg_str = arg_reg.encode(size=clause.size)
			output(f'mov {reg_str}, {clause.asm_str.format(dest_reg=arg_reg)}')

		arg_types.append(T)

	if fn_header is ALLOC_FN:
		if not arg_types:
			arg_types.append(UNSPECIFIED_INT)
			output(f'mov {arg_reg:8}, {alloc_fac}')
		# return call_function_direct(fn_header, arg_types, variables=variables, caller_type=caller_type)
		
	# use fn_header.typeargs, fn_header.args
	# We could store {typename: Type}, but rn we have {typename: typename}

	if len(arg_types) != len(fn_header.args):
		fl = len(fn_header.args)
		al = len(arg_types)

		err(f'{fn_header!r} expects exactly '
			f'{fl} argument{"s" * (fl != 1)}, '
			f'but {al} {"were" if al != 1 else "was"} provided')

	# Populate type_mappings
	type_mappings = {T: None for T in fn_header.typeargs}
	if caller_type is not None:
		type_mappings |= dict(
			zip(caller_type.parent.args, caller_type.args)
		)
	# print('TYPE MAPPING USING', fn_header.args, 'AND', arg_types)
	for i, ((type_str, arg_name), arg_type) in enumerate(zip(fn_header.args, arg_types), 1):

		if arg_type is not UNSPECIFIED_INT:
			curr_mappings = arg_type.match_pattern(
				type_str, fn_header.module
			)
		else:
			parse_type_result = parse_type(
				type_str, fn_header.module.children, variables=variables
			)

			if isinstance(parse_type_result, ParseTypeError):
				# Trying to match against a polymorphic type

				if len(type_str.split(maxsplit=1)) > 1:
					err('Integer literal cannot be matched to type '
						f'{type_str.lstrip()!r}')

				curr_mappings = {type_str.lstrip(): UNSPECIFIED_INT}
			else:
				# concrete type, no mappings
				T, = parse_type_result
				if not T.is_int():
					err(f'Provided an integer as argument, but expected {T}')
				# print('MAPPED, UNSPECIFIED')
				continue

		for type_arg, proposed_mapping in curr_mappings.items():
			if type_arg not in type_mappings:
				err(f'{type_arg!r} in {type_str} is neither '
					'an existing type nor a type argument')

			existing_mapping = type_mappings[type_arg]

			if existing_mapping is None:
				type_mappings[type_arg] = proposed_mapping
				continue
			if existing_mapping is ANY_TYPE: continue
			if proposed_mapping is ANY_TYPE:
				type_mappings[type_arg] = proposed_mapping
				continue

			# Assign a more specific int type
			if existing_mapping is UNSPECIFIED_INT:
				if proposed_mapping.is_int():
					type_mappings[type_arg] = proposed_mapping
					continue

			# Don't do anything if specific already exists
			elif proposed_mapping is UNSPECIFIED_INT:
				if existing_mapping.is_int(): continue

			elif proposed_mapping is existing_mapping: continue

			# Type matching failed
			err('Multiple mappings to same type argument. '
				f'Trying to map {type_arg!r} to '
				f'{proposed_mapping} and {existing_mapping} '
				f'{arg_types}')

	# Handles UNSPECIFIED_INT
	for typename, subbed_type in type_mappings.items():
		if subbed_type is UNSPECIFIED_INT:
			type_mappings[typename] = INT_TYPE

	try:
		instance_key = tuple(type_mappings[typearg_name].name for typearg_name in fn_header.typeargs)
	except AttributeError:
		for typearg_name in fn_header.typeargs:
			if type_mappings[typearg_name] is not None: continue
			err(f'Type argument {typearg_name!r} not mapped in {fn_header!r}')
		err('[Internal Error] All types mapped but still got a TypeError')

	if instance_key in fn_header.instances:
		# print('Queued and done', (fn_header, instance_key))
		fn_instance = fn_header.instances[instance_key]
	else:
		# print('Adding to queue', (fn_header, instance_key))
		fn_queue.append((fn_header, instance_key))
		fn_instance = fn_header.add_sub(
			instance_key, (*type_mappings.values(),)
		)

	output('call', fn_instance.mangle())

	if fn_header is ALLOC_FN:
		return alloc_type.pointer()

	# print(f'{fn_header.ret_type = }')

	parse_type_result = parse_type(
		fn_header.ret_type,
		fn_header.module.children | type_mappings,
		variables = variables,
	)
	if isinstance(parse_type_result, ParseTypeError):
		err(f'In {fn_header.ret_type!r}, {parse_type_result}')
	if len(parse_type_result) != 1:
		err('Return type string does not evaluate to a single type')
	ret_type, = parse_type_result

	return ret_type

def get_operator_insts(operator, operand_clause, operand_type):
	print(f'{Shared.line_no}: OPERATION {operator} USING {operand_type} (size = {operand_clause.size})')
	if Type.get_size(operand_type) != operand_clause.size:
		err('[INTERNAL ERROR] clause and type size do not match')
	# Should not muddle registers. So we can't call functions.

	# I don't put any constraints. Makes it unsafe, but also flexible.
	# We'll see if that's a good idea.

	if   operator == '+': inst = 'add'
	elif operator == '-': inst = 'sub'
	elif operator == '&': inst = 'and'
	elif operator == '^': inst = 'xor'
	elif operator == '|': inst = 'or'
	elif operator in ('<', '>', '<=', '>='): inst = 'cmp'
	elif operator in ('==', '!='):
		# TODO: get_operator_insts() should take type of both operands

		if operand_type is STR_TYPE:
			return [
				f'mov {arg_regs[0]:8}, {{dest_reg:8}}',
				f'mov {arg_regs[1]:{operand_clause.size}}, {operand_clause.asm_str}',
				'call strcmp',
				'cmp al, 0'
			]
		inst = 'cmp'
	elif operator == '<<':
		size = Type.get_size(operand_type)
		return [
			f'mov {Register.c:{operand_clause.size}}, {operand_clause.asm_str}',
			f'shl {{dest_reg:{size}}}, cl',
		]
	elif operator == '>>':
		size = Type.get_size(operand_type)
		return [
			f'mov {Register.c:{operand_clause.size}}, {operand_clause.asm_str}',
			f'shr {{dest_reg:{size}}}, cl',
		]
	elif operator == '*':
		size = Type.get_size(operand_type)
		return [
			f'mov {Register.a:{size}}, {{dest_reg:{size}}}',
			f'xor rdx, rdx',
			f'mov {Register.b:{operand_clause.size}}, {operand_clause.asm_str}',
			f'mul {Register.b:{size}}',
			f'mov {{dest_reg:{size}}}, {Register.a:{size}}'
		]
	elif operator == '/':
		size = Type.get_size(operand_type)
		return [
			f'mov {Register.a:{size}}, {{dest_reg:{size}}}',
			f'xor rdx, rdx',
			f'mov {Register.b:{operand_clause.size}}, {operand_clause.asm_str}',
			f'div {Register.b:{size}}',
			f'mov {{dest_reg:{size}}}, {Register.a:{size}}'
		]
	elif operator == '%':
		size = Type.get_size(operand_type)
		return [
			f'mov {Register.a:{size}}, {{dest_reg:{size}}}',
			f'xor rdx, rdx',
			f'mov {Register.b:{operand_clause.size}}, {operand_clause.asm_str}',
			f'div {Register.b:{operand_clause.size}}',
			f'mov {{dest_reg:{size}}}, {Register.d:{size}}'
		]
	else:
		# others?
		err(f'Operator {operator} not supported')

	return [f'{inst} {{dest_reg:{operand_clause.size}}}, {operand_clause.asm_str}']

def operator_result_type(operator, l_type, r_type) -> Type:
	if operator == '-':
		# ptr - ptr, ptr - int, int - int
		if l_type.deref is not None:
			if r_type.is_int(): return l_type
			if r_type.deref is not None: return UNSPECIFIED_INT
			err(f'Cannot subtract {r_type} from a pointer')

		if l_type.is_int() and r_type.is_int():
			if l_type is not r_type: return UNSPECIFIED_INT
			return l_type

		err(f'Cannot subtract between {l_type} and {r_type}')

	if operator in ('+', '-'):
		# ptr + int, int + ptr, int + int
		if l_type.is_int():
			i_type = l_type
			o_type = r_type
		elif r_type.is_int():
			i_type = r_type
			o_type = l_type
		else:
			err(f'Cannot add {l_type} and {r_type}')

		if o_type.deref is not None:
			return o_type

		if not o_type.is_int():
			err(f'Cannot add {l_type} and {r_type}')

		if l_type is not r_type: return UNSPECIFIED_INT
		return l_type

	if operator == '==': return Flag.e
	if operator == '!=': return Flag.ne

	if l_type != r_type:
		if not (l_type.is_int() and r_type.is_int()):
			err(f'Unsupported operator {operator!r} '
				f'between {l_type} and {r_type}')

	if operator == '>':  return Flag.g
	if operator == '<':  return Flag.l
	if operator == '>=': return Flag.ge
	if operator == '<=': return Flag.le

	if l_type.is_int() and r_type.is_int():
		if r_type is not l_type: return UNSPECIFIED_INT
		return l_type

	err(f'Unsupported operator {operator!r} between {l_type} and {r_type}')

def get_string_label(string, strings):
	if string in strings: return strings[string]
	label = f'_s{len(strings)}'
	strings[string] = label
	return label

def split_size(size) -> tuple[int, ...]:
	sizes = []
	if size&0b001: sizes.append(0b001)
	if size&0b010: sizes.append(0b010)
	if size&0b100: sizes.append(0b100)

	eights = size >> 3
	sizes += [8] * eights

	return sizes

def get_discriminator_size(max_field_id):
	if max_field_id is None: return 0
	return (max_field_id.bit_length()-1) // 8 + 1

# Strongly typed

if Shared.arch is Arch.win64:
	arg_regs = (Register.c, Register.d, Register.r8, Register.r9)
else:
	arg_regs = (Register.di, Register.si, Register.c, Register.d, Register.r8, Register.r9)

standard_dest_regs = (Register.a, Register.b)
dest_reg_fmts = ('dest_reg', 'aux_reg')

orange_dir = os.path.dirname(__file__)

if __name__ == '__main__':
	core_file = open(f'{orange_dir}/lib/core.or')
	std_file = open(f'{orange_dir}/lib/std.or')

	# Builtins

	PTR_TYPE = Type('_Ptr', PTR_SIZE, args=('T',))
	PTR_TYPE.size = 8

	types = {}
	core_module = None

	Shared.infile = core_file
	core_module = Type.read_module('_core', in_module=False)

	builtin_types = core_module.children

	CORE_PTR_TYPE = core_module.children['_Ptr']

	# modify the dict so it effects existing instances
	PTR_TYPE.methods |= CORE_PTR_TYPE.methods

	builtin_types['any'] = Type('any', None)
	ANY_TYPE = builtin_types['any']

	U64_TYPE = builtin_types['u64']
	U64_TYPE.size = 8
	# print('FORCE SET SIZE = 8 [U64]')

	VOID_TYPE = builtin_types['void']
	VOID_TYPE.size = 0
	# print('FORCE SET SIZE = 0 [VOID]')

	CHAR_TYPE = builtin_types['char']
	CHAR_TYPE.size = 1
	# print('FORCE SET SIZE = 1 [CHAR]')

	INT_TYPE = builtin_types['int']
	INT_TYPE.size = 4
	# print('FORCE SET SIZE = 4 [INT]')

	STR_TYPE = builtin_types['str']
	STR_TYPE.deref = CHAR_TYPE
	STR_TYPE.consts['null'].type = STR_TYPE

	# print('STR_TYPE size =', STR_TYPE.size)

	builtin_type_set = {*builtin_types.values(), UNSPECIFIED_INT, FLAG_TYPE}

	Shared.infile = std_file
	std_module = Type.read_module('_std', in_module=False)

	core_module.methods |= std_module.methods
	core_module.children |= std_module.children
	core_module.consts |= std_module.consts

	main_dir = os.path.dirname(os.path.abspath(argv[1]))
	os.chdir(main_dir)

	print()

	fn_queue = []

	Shared.infile = arg_infile
	main_module = Type.read_module('_main', in_module=False)
	print()

	if 'main' not in main_module.methods:
		err("No definition of function 'main' found.")

	main_header = main_module.methods['main']
	if () in main_header.instances:
		fn_instance = main_header.instances[()]
	else:
		output('global main')
		fn_queue.append((main_header, ()))
		fn_instance = main_header.add_sub(())

	strings = {}

	while fn_queue:
		fn, instance_key = fn_queue.pop(0)
		fn_instance = fn.instances[instance_key]

		# print('\n', fn.name, instance_key, sep = '')

		if fn_instance.template.isextern:
			# print(f'DEQUEUED EXTERN {fn}, {instance_key}')
			continue
		# else:
			# print('DEQUEUED Function', fn)

		# fn_instance = fn.add_sub(instance_key)
		# if fn_instance is None: continue

		# 2 passes. allocate variable space first

		output(f'\n; {fn_instance.type_mappings}')

		# print('INSTANTIATE INSTANCE', fn.name, instance_key, fn_instance.mangle())

		curr_mod = fn.module
		types = curr_mod.children

		fn_types = types | fn_instance.type_mappings

		fn_instance.init_args_vars(types)
		offset = fn_instance.offset

		variables = curr_mod.consts | fn_instance.arg_vars
		local_variables = {*fn_instance.arg_vars}

		# print('Instantiating new concrete function:')
		# print('Instantiating', fn.name, 'with', fn_instance.type_mappings, 'as', fn_instance.mangle())
		# print('Module:', curr_mod)
		# print('Variables:', variables)
		# print('Constants:', curr_mod.consts)
		# print('Types:', fn_types)

		scope_level = 1

		Shared.infile = fn.infile

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
				# print('DECLARATION:', (name, type_str))

				if name in local_variables:
					var = variables[name]
					err(f'Variable {name!r} already declared in '
						f'line {var.decl_line_no}')

				# print(f'{type_str = }')
				parse_type_result = parse_type(type_str.lstrip(), fn_types,
					variables=variables)
				# print('EXIT PARSETYPE')
				if isinstance(parse_type_result, ParseTypeError):
					err(f'[in {type_str.lstrip()!r}] {parse_type_result}')
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

				# print(f'Declaration type list of {type_str!r}: {parse_type_result}')
				T, = parse_type_result

				if T is ANY_TYPE:
					err("A variable of type 'any' must be a pointer")

				offset += T.size
				local_variables.add(name)
				variables[name] = Variable(name, offset, T, Shared.line_no)

			elif match[1] == 'const':
				split = match[2].split(maxsplit=1)
				if len(split) != 2:
					err('An expression is required to declare a constant')

				name, exp = split

				if name in local_variables:
					err(f'{name!r} is already declared')

				local_variables.add(name)
				const_var = eval_const(exp, fn_types,
					variables=variables)
				# const_var.name = name
				variables[name] = const_var


			elif match[1] in ('if', 'while'):
				scope_level += 1

			elif match[1] == 'end':
				scope_level -= 1
				if not scope_level: break  # end of function


		# I might be able to support overloading too and just disallow conflicts
		# We already have the line number in the struct, so we can error nicely

		# Code gen
		output(f'{fn_instance.mangle()}:')

		output('push rbp')
		output(f'mov rbp, rsp')  # 32 extra bytes are always required

		# align and push only if there are function calls
		offset = ((offset+1) | 15) + 33  # (round up to multiple of 15) + 32
		output(f'sub rsp, {offset}')

		for var in variables.values():
			if isinstance(var, Const): continue
			var.offset = offset - var.offset

		# Populate arguments
		if len(fn.args) > len(arg_regs):
			err('[Internal Error] Too many arguments; this was not checked earlier')
		for (_, argname), arg_reg in zip(fn.args, arg_regs):
			arg = variables[argname]
			reg_str = arg_reg.encode(size=arg.type.size)
			output(f'mov {size_prefix(arg.type.size)} [rsp + {arg.offset}], ',
				reg_str)

		ctrl_stack = [Ctrl(0, Branch.FUNCTION)]

		Shared.infile.seek(fn.tell)
		for Shared.line_no, Shared.line in enumerate(Shared.infile, fn.line_no+1):

			match = Patterns.stmt.match(Shared.line)
			line = match[2]  # maybe indentation?
			if not line: continue

			output(f'; ({Shared.line_no}) {Shared.line.strip()}')
			# print(f'{Shared.line_no} {Shared.line.strip()!r}')

			match = Patterns.split_word.match(line)

			if not match: match = Subscriptable(); print(match)

			if   match[1] == 'let': continue
			elif match[1] == 'const': continue
			elif match[1] == 'return':
				if fn.name == 'main': dest_regs = arg_regs[0],
				else: dest_regs = standard_dest_regs

				if not match[2]:
					ret_type = types['void']
				else:
					# We don't use the expected size
					# for the case of returning UNSPECIFIED_INT
					ret_type = parse_exp(match[2].strip(),
						dest_regs = dest_regs, fn_queue = fn_queue,
						variables = variables)

				fn_type_list = parse_type(fn.ret_type, fn_types,
					variables=variables)

				if fn_type_list is None:
					# TODO: better message
					err(f'Type {fn.ret_type!r} is not available.')
				if len(fn_type_list) != 1:
					err('Return type must be exactly one type')
				expected_ret_type, = fn_type_list

				if ret_type is UNSPECIFIED_INT:
					if not expected_ret_type.is_int():
						err('Mismatched type. '
							f'{fn.name} expects {expected_ret_type}. '
							f'Trying to return {ret_type}')
				elif ret_type is not expected_ret_type:
					err('Mismatched type. '
						f'{fn.name} expects {expected_ret_type}. '
						f'Trying to return {ret_type}')

				if fn.name == 'main':
					output('call exit')
				else:
					output('mov rsp, rbp')
					output('pop rbp')
					output('ret')

			elif match[1] == 'while':
				ctrl_no = Ctrl.next()

				ctrl = Ctrl(ctrl_no, Branch.WHILE)
				output(f'{ctrl.label}:')
				ctrl_stack.append(ctrl)

				ret_flag = parse_exp(match[2].strip(),
					dest_regs = Flag, fn_queue = fn_queue,
					variables = variables
				)
				if ret_flag is Flag.NEVER:
					output(f'jmp _E{ctrl_no}')
				elif ret_flag is not Flag.ALWAYS:
					output(f'j{(~ret_flag).name} _E{ctrl_no}')

			elif match[1] == 'if':
				ctrl_no = Ctrl.next()
				ctrl = Ctrl(ctrl_no, 0)
				ctrl_stack.append(ctrl)

				# TODO: parse_exp() returns a Flag object if dest_reg is flags
				ret_flag = parse_exp(match[2].strip(),
					dest_regs = Flag, fn_queue = fn_queue,
					variables = variables
				)

				if ret_flag is Flag.NEVER:
					output(f'jmp _E{ctrl_no}_1')
				elif ret_flag is not Flag.ALWAYS:
					output(f'j{(~ret_flag).name} _E{ctrl_no}_1')

			elif match[1] == 'elif':
				if not ctrl_stack or ctrl_stack[-1].branch is Branch.WHILE:
					err('elif is not after if')

				ctrl = ctrl_stack[-1]

				ctrl.branch += 1

				output(f'jmp _END{ctrl.ctrl_no}')
				output(f'_E{ctrl.ctrl_no}_{ctrl.branch}:')


				# TODO: parse_exp() returns a Flag object if dest_reg is flags
				ret_flag = parse_exp(match[2].strip(),
					dest_regs = Flag, fn_queue = fn_queue,
					variables = variables
				)

				if ret_flag is Flag.NEVER:
					output(f'jmp _E{ctrl.ctrl_no}_{ctrl.branch+1}')
				elif ret_flag is not Flag.ALWAYS:
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

			elif match[1] == 'enum':
				err('Local types are not yet supported')

			elif match[1] == 'import':
				err('Cannot import inside a function')

			elif match[1] == 'export':
				err('Cannot export inside a function')

			else:
				match = Patterns.through_strings(r'(?<!=)=(?!=)').match(line)

				if match:
					exp = match['post'].strip()
					dest = match['pre'].strip()
					# print(f'{exp = }; {dest = }; {match[2] = }')
				else:
					exp = line
					dest = None

				ret_type = parse_exp(exp.strip(),
					dest_regs = standard_dest_regs, fn_queue = fn_queue,
					variables = variables)

				if dest is not None:
					index = Patterns.find_through_strings(dest, '[')

					if index != -1:
						dest_token = dest[:index]
						args_str = dest[index:].strip()
					else:
						dest_token = dest

					insts, dest_clauses, dest_type = parse_token(
						dest_token, fn_types, variables = variables
					)

					if dest_type is UNSPECIFIED_INT or not dest_clauses:
						err(f'Cannot assign to {dest}')

					if index != -1:
						first_arg = arg_regs[0]
						second_arg = arg_regs[1]

						if len(dest_clauses) != 1:
							err(f"Item assignment does not support non-standard sizes yet ({len(dest_clauses)} clauses)")

						dest_clause, = dest_clauses

						# the value of the expression is in rax
						for inst in insts:
							output(inst.format(dest_reg=first_arg))

						# TODO: Use big moves
						# TODO: I need a convenience function
						# so I can use it without thinking
						# and I don't make mistakes.
						# But first, I need to figure out
						# what it means to not make a mistake.

						# print(f'{dest_clause = }')
						output(
							f'mov {first_arg:{dest_clause.size}}, '
							f'''{dest_clause.asm_str.format(
								dest_reg=first_arg, aux_reg=second_arg
							)}'''
						)
						# print(f'Moving into {dest_clause!r}')
						ret_size = Type.get_size(ret_type)
						output(f'mov {second_arg:{ret_size}}, '
							f'{Register.a:{ret_size}}')

						if '_setitem' in dest_type.methods:
							caller_type = dest_type
						else:
							caller_type = dest_type.deref
							if caller_type is None or '_setitem' not in caller_type.methods:
								err(f'{dest_type} does not support item assignment')

						fn_header = caller_type.methods['_setitem']

						setitem_result = call_function(fn_header,
							[dest_type, ret_type], args_str,
							variables=variables)

						if setitem_result is not VOID_TYPE:
							err('_setitem() must return void')

					else:
						if ret_type is UNSPECIFIED_INT:
							if not dest_type.is_int():
								err(f'Cannot assign an integer into '
									f'variable {dest} of {dest_type}')
						elif ret_type is not dest_type:
							err(f'Cannot assign {ret_type} into '
								f'variable {dest} of {dest_type}')

						# IMPORTANT: Base assignment logic

						# the value of the expression is in rax
						for inst in insts:
							output(
								inst.format(dest_reg=Register.c, aux_reg=Register.d)
							)

						# print(f'Moving into {dest_clauses!r} using {insts}')
						for dest_reg, dest_clause in zip(
							standard_dest_regs,
							dest_clauses,
						):
							output(
								f'''mov {dest_clause.asm_str.format(
									dest_reg=arg_regs[0], aux_reg=arg_regs[1]
								)}, '''
								f'{dest_reg:{dest_clause.size}}'
							)

				output()

		# which registers need to be preserved now?
		if fn.name == 'main':
			output(f'xor {arg_regs[0]:8}, {arg_regs[0]:8}')
			output('call exit')
		else:
			output('mov rsp, rbp')
			output('pop rbp')
			output('ret')

		# print()

	output(r'''
	segment .data
	''')

	for string, label in strings.items():
		encoded_string = repr(string)[2:-1].replace('`', '\\`')
		output(f'{label}: db `{encoded_string}`, 0')

	Shared.out.close()

	commands = []

	if Shared.assemble:
		commands.append(f"nasm \"{Shared.out.name}\" -f {Shared.arch.name} -o \"{file_name}.o\"")

	if Shared.link:
		if Shared.debug:
			linker = 'gcc-asm'
		else:
			linker = 'gcc'

		if Shared.arch is Arch.win64:
			bin_extension = '.exe'
		elif Shared.arch is Arch.elf64:
			bin_extension = ''
		else:
			raise TypeError(f'Unsupported architecture {Shared.arch!r}')

		commands.append(
			f'{linker} "{file_name}.o" -o "{file_name}{bin_extension}"'

			# NOTE: prone to injection
			+ ''.join(f' -L{library}' for library in Shared.library_paths)
			+ ''.join(f' -l{library}' for library in Shared.libraries)
		)

	if commands:
		cmd = ' && '.join(commands)
		print('running:\n', cmd)
		result = system(cmd)
