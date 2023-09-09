# poly _(name subject to change)_

poly is a strongly typed compiled language with polymorphic functions and structs.

## Basic Syntax

```ts
// 'type' is just a fancy word for 'struct'
type Int_node
    // members of the type are declared using 'let'
    let data int
    let next &Int_node  // we can do self referential members!
end

// arguments start after the ':'
fn int_list_len: &Int_node node -> int
    // variable declarations have the same syntax as type fields
    let len int
    len = 0
    while node != 0
        node = node.next  // implicit dereference
        len = len + 1
    end
    return len
end

// main is the entry point
// also, arguments and return are optional
fn main
    let node &Int_node

    // alloc is a builtin function that returns a pointer to that type
    node = alloc(Int_node)

    // node = alloc(int)  // this wouldn't work because we're strongly typed

    println(node.data)

    let len int
    len = int_list_len(node)
    println(len)
end
```

## Usage

Clone the repository to be able to use the language or [download the repo files](https://github.com/cubed-guy/poly/archive/refs/heads/master.zip).

You'll also need to get [nasm](https://www.nasm.us/pub/nasm/snapshots/latest/) and gcc.

**Note:** _For Windows, use [this direct link](https://sourceforge.net/projects/mingw-w64/files/Toolchains%20targetting%20Win64/Personal%20Builds/mingw-builds/8.1.0/threads-posix/seh/x86_64-8.1.0-release-posix-seh-rt_v6-rev0.7z) to download the gcc version that works for poly._

**Compile a `.poly` file on Windows**
```batch
> python3 <path_to_compiler>/polymorphism.py <poly_file_name>.poly <assembly_file_name>.asm
> <path to nasm> -fwin64 <assembly_file_name>.asm -o <object_file_name>.o
> <path to gcc> <object_file_name>.o -o <executable_file_name>.exe
```
`nasm` and `gcc` are found in their corresponding `bin` folders. Alternatively, you can add the `bin` folders to your path variables to avoid typing the path a every time.

**Compile a `.poly` file on Linux (and perhaps macOS)**
```bash
$ python3 <path_to_compiler>/polymorphism.py <poly_file_name>.poly <assembly_file_name>.asm
$ nasm -felf64 <assembly_file_name>.asm -o <object_file_name>.o
$ gcc -no-pie <object_file_name>.o -o <binary_file_name>
```

## Polymorphism
```ts
// types can take type arguments
type Node T
    let data T

    // you can pass in arguments to types while declaring variables
    let next &Node T
end

// functions can also take type arguments
fn list_len T: &Node T node -> int
    let len int
    len = 0
    while node != 0
        node = node.next
        len = len + 1
    end
    return len
end

fn main
    let char_node &Node char  // we have instantiated 'Node' with 'char'
    let int_node &Node int

    char_node = alloc(Node char)
    int_node = alloc(Node int)

    let int_list_len int
    let char_list_len int

    // list_len works for both Node T
    char_list_len = list_len(char_node)
    int_list_len = list_len(int_node)

    // list_len(5)  // this would fail because 5 is does not satisfy '&Node T'

    println(char_list_len)
    println(int_list_len)
end
```
