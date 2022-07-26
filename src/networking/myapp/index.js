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