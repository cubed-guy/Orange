# Orange

Orange is a strongly typed compiled language with polymorphic functions and structs.

## Example

_Check out [example.or](example.or) for a more thorough example._

```ts
// A simple program demonstrating polymorphism in Orange

type Node T  // This is a polymorphic type. It takes a type as an argument at compile-time.
    let data T
    let next &Node T
end

fn main
    // using the polymorphic type with different arguments
    let i_node &Node int
    let c_node &Node char

    // dynamic memory allocation
    i_node = alloc(Node int)
    c_node = alloc(Node char)

    // the dot '.' operator can be used on pointers
    i_node.data = 5
    c_node.data = 'E'

    // println is a polymorphic function that can take an argument of any type
    println(i_node.data)  // 5
    println(c_node.data)  // E
end
```

## Setup

1. Clone or [download](https://github.com/cubed-guy/orange/archive/refs/heads/master.zip) the repository.
2. You'll also need to get [nasm](https://www.nasm.us/pub/nasm/snapshots/latest/) and gcc.
3. After installation `nasm` and `gcc` are found in their corresponding `bin` folders. Orange can compile your programs correctly only if you add the `bin` folders [to your PATH variable](https://stackoverflow.com/a/44272417/10826013).

**Note:** _For Windows, use [this direct link](https://sourceforge.net/projects/mingw-w64/files/Toolchains%20targetting%20Win64/Personal%20Builds/mingw-builds/8.1.0/threads-posix/seh/x86_64-8.1.0-release-posix-seh-rt_v6-rev0.7z) to download the gcc version that works for Orange. (It requires 7zip, so make sure you have that too, or convert it online)_

## Compiling a Program Written in Orange

```batch
> python3 orange.py <orange_source_file>.or <output_file_name>
```
_Make sure you substitute `<orange_source_file>` and `<output_file_name>` with the names of your files._

This will convert your program written in Orange into an executable binary that you can run.

The output file name is optional. If you don't provide it, the output binary will have the name based on the input file.

## Basic Syntax

_A detailed look at the language can be found [in the wiki](../../wiki)._

```ts
// 'type' is just a fancy word for 'struct'
type Int_node
    // members of the type are declared using 'let'
    let data int
    let next &Int_node  // we can do self-referential members!
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

## Polymorphism

The code snippet above defines a linked list for just integers. The function to find the length and any other functions that you would define for it would be stuck to only the integer variant of linked lists. If you wanted a linked list of characters now, you would have to write them again, but this time for characters.

Instead, we will let the compiler do the work for us. With the power of **polymorphism** in Orange, we need to define the linked list node type and its functions only once, but then use it to create linked lists of integers as well as those of characters.

```ts
// types can take type arguments
type Node T
    let data T
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
    let char_node &Node char  // we have created a 'Node' of 'char'
    let int_node  &Node int   // we have created a 'Node' of 'int'

    char_node = alloc(Node char)
    int_node  = alloc(Node int)

    let int_list_len int
    let char_list_len int

    // list_len works for both Node T
    char_list_len = list_len(char_node)
    int_list_len = list_len(int_node)

    println(char_list_len)
    println(int_list_len)
end
```
