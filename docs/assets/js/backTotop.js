// https://css-tricks.com/debouncing-throttling-explained-examples/
// https://stackoverflow.com/questions/31223341/detecting-scroll-direction
let scroll_position = 0;
let scroll_direction;
let scroll_up_count = 0;

window.addEventListener('scroll', function(e){
  if ((document.body.getBoundingClientRect()).top > scroll_position) {
    scroll_up_count ++;
  }
  scroll_position = (document.body.getBoundingClientRect()).top;

  let back_to_top = document.getElementById("back-top");
  if (scroll_up_count > 5) {
    this.setTimeout(()=>{
      back_to_top.removeAttribute("hidden");
      console.log(scroll_up_count);
      scroll_up_count = 0;
    }, "1000")
  }
  back_to_top.setAttribute("hidden", "hidden");
});

