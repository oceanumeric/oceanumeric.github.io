# Web Server and REST API

In previous posts, we have studied how to create a web server by using the
`socket` package in `Python`. However, our simple server could only response to
one request one time when it was listening. Or loosely speaking, it is 
asynchronous. In this post, we will learn more about creating web server and 
asynchronous programming with `JavaScript`. 


## Blocking I/O examples

Input/output (IO) refers to interaction with devices such as a hard drive, 
network or database. Generally anything that is not happening in the CPU 
is called IO. When you call an API that requests data from IO, you will 
not get a response instantly, but with some delay. However, during this process,
you CPU is still running. You could either make your CPU idle and waiting for 
the response or keep it busy for other operations. 

=== "sever.py"

    ```py
    import socket


    def main() -> None:
        host = socket.gethostname()
        port = 12000
        
        # create a TCP/IP socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            # bind the socket to the port
            sock.bind((host, port))
            # listen for incoming connections
            sock.listen(5)
            print("Server started...")
            # accepting the incoming connection, blocking
            conn, addr = sock.accept()  
            print('Connected by ' + str(addr))
            while True:
                data = conn.recv(1024)  # receving data, blocking
                if not data: 
                    break
                print(data)
                modified_message = data.decode().upper()
                conn.send(modified_message.encode())
                    

    if __name__ == "__main__":
        main()
    ```


=== "client.py"

    ```py
    import socket
    import sys
    import time


    def main() -> None:
        host = socket.gethostname()
        port = 12000

        # create a TCP/IP socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            while True:
                sock.connect((host, port))
                while True:
                    data = str.encode(sys.argv[1])
                    sock.send(data)
                    time.sleep(0.5)
                    

    if __name__ == "__main__":
        assert len(sys.argv) > 1, "Please provide message"
        main() 
    ```

If you tried the above programming, you can find that the sever could not 
receive another message from client. To enable a server to handle thousands of
concurrent connections, you could either use _thread concurrency_ or
_non-block I/O_. 


## `Node.js` and non-blocking I/O


`Node.js` was created to make non-blocking I/O possible. The creator of `Node.js`
- Ryan Dahl, explains this idea very well in the following talk. 

<iframe width="560" height="315" src="https://www.youtube.com/embed/M-sc73Y-zQA?start=425" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


When `Node.js` performs an I/O operation, like reading from the network, 
accessing a database or the filesystem, instead of blocking the thread and 
wasting CPU cycles waiting, `Node.js` will resume the operations when the 
response comes back.

This allows Node.js to handle thousands of _concurrent connections_ with a 
single server without introducing the burden of managing _thread concurrency_, 
which could be a significant source of bugs.

## Event loop

Non-blocking I/O is built upon the event loop which is quite unique runtime
environment in `web` and `javascript`. The runtime environment has two stacks:

* `call` stack
* message queue 

The event loop gives priority to the call stack, and it first processes 
everything it finds in the call stack, and once there's nothing in there, 
it goes to pick up things in the message queue.

## Create a sever with `Node.js`

```js
const http = require('http')
const port = 3001


let notes = [
    {
      id: 1,
      content: "HTML is easy",
      date: "2022-05-30T17:30:31.098Z",
      important: true
    },
    {
      id: 2,
      content: "Browser can execute only Javascript",
      date: "2022-05-30T18:39:34.091Z",
      important: false
    },
    {
      id: 3,
      content: "GET and POST are the most important methods of HTTP protocol",
      date: "2022-05-30T19:20:14.298Z",
      important: true
    }
  ]

const app = http.createServer((request, response) => {
    response.writeHead(200, { 'Content-Type': 'application/json' })
    response.end(JSON.stringify(notes))
})

app.listen(port)

console.log(`Server running on port ${port}`)
```

## Create a server with `Express.js`

```js
// create a web server with express
const { response, request } = require("express")
const express = require("express")
const app = express()

let notes = [
    {
      id: 1,
      content: "HTML is easy",
      date: "2022-05-30T17:30:31.098Z",
      important: true
    },
    {
      id: 2,
      content: "Browser can execute only Javascript",
      date: "2022-05-30T18:39:34.091Z",
      important: false
    },
    {
      id: 3,
      content: "GET and POST are the most important methods of HTTP protocol",
      date: "2022-05-30T19:20:14.298Z",
      important: true
    }
  ]

// root directory

app.get('/', (request, response) => {
    response.send("<h1>Hello World!</h1>")
})

app.get('/api/notes', (request, response) => {
    response.json(notes)
})

const port = 3000

app.listen(port, () => {
    console.log(`Server running on port ${port}`)
})
```

## REST

Let's expand our application so that it provides the same RESTful HTTP API as json-server.

Representational State Transfer, aka REST, was introduced in 2000 in Roy 
Fielding's dissertation. REST is an architectural style meant for building 
scalable web applications.

We are not going to dig into Fielding's definition of REST or spend time 
pondering about what is and isn't RESTful. Instead, we take a more narrow 
view by only concerning ourselves with how RESTful APIs are typically 
understood in web applications. The original definition of REST is in fact 
not even limited to web applications.

We mentioned in the previous part that singular things, like notes in the case 
of our application, are called resources in RESTful thinking. Every resource 
has an associated URL which is the resource's unique address.

We can execute different operations on resources. The operation to be executed 
is defined by the HTTP _verb_:

| URL      | verb   | functionality                                                    |
|----------|--------|------------------------------------------------------------------|
| notes/10 | GET    | fetches a single resource                                        |
| notes    | GET    | fetches all resources in the collection                          |
| notes    | POST   | creates a new resource based on the request data                 |
| notes/10 | DELETE | removes the identified resource                                  |
| notes/10 | PUT    | replaces the entire identified resource with the request data    |
| notes/10 | PATCH  | replaces a part of the identified resource with the request data |


```js
// create a web server with express
const { response, request } = require("express")
const express = require("express")
const app = express()
app.use(express.json())  // activate express json parser 

let notes = [
    {
      id: 1,
      content: "HTML is easy",
      date: "2022-05-30T17:30:31.098Z",
      important: true
    },
    {
      id: 2,
      content: "Browser can execute only Javascript",
      date: "2022-05-30T18:39:34.091Z",
      important: false
    },
    {
      id: 3,
      content: "GET and POST are the most important methods of HTTP protocol",
      date: "2022-05-30T19:20:14.298Z",
      important: true
    }
  ]

// root directory
app.get('/', (request, response) => {
    response.send("<h1>Hello World!</h1>")
})

// single resource :id as parameter 
app.get('/api/notes/:id', (request, response) => {
    const id = Number(request.params.id)
    console.log(id)
    const note = notes.find(note => note.id === id)
    if (note) {
        console.log(note)
        response.json(note)
    } else {
        response.status(404).end()
    }
})

// delete
app.delete('/api/notes/:id', (request, response) => {
    const id = Number(request.params.id)
    notes = notes.filter(note => note.id !== id)

    response.status(204).end()
})

// post
app.post('/api/notes', (request, response) => {
    const note = request.body
    console.log(note)
    response.json(note)
})

const port = 3000

app.listen(port, () => {
    console.log(`Server running on port ${port}`)
})
```