const C = ObjC.classes.HuntWord;
const regex = /(word|create|game|start)/i;

console.log("[ methods ]");
C.$methods
	.filter(m => regex.test(m))
	.sort()
	.forEach(m => console.log(m));

//console.log("\n[ HuntWord ivars ]");
//C.$ivars.forEach(i => console.log(i));
