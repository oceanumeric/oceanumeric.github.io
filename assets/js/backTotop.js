// https://css-tricks.com/debouncing-throttling-explained-examples/
// https://stackoverflow.com/questions/31223341/detecting-scroll-direction
var scroll_position = 0;
var scroll_direction;
var i = 0;
var temp = 0;

let back_to_top = document.getElementById("back-top");


window.addEventListener('scroll', function(e){
  setTimeout(() => {
    if (
      (document.body.getBoundingClientRect()).top > scroll_position
      ) {
        i++;
      }
    if (i > temp) {
      setTimeout(()=>{
        back_to_top.removeAttribute("hidden");
      })
    }
    temp = i;
    scroll_position = (document.body.getBoundingClientRect()).top;
    back_to_top.setAttribute("hidden", "hidden");
  })
});



