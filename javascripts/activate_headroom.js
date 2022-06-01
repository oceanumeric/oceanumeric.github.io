document.addEventListener("DOMContentLoaded", function(event) {
    var headerElement = document.querySelector("header");

    var headroom = new Headroom(headerElement, {
      "offset": 205,
      "tolerance": {
          "up" : 40,
          "down" : 0
      },
      "classes": {
        "initial": "animated",
        "pinned": "slideDown",
        "unpinned": "slideUp"
      }
    });
    headroom.init();
});