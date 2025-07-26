const className = "HuntWord";

if (ObjC.available) {
   const cls = ObjC.classes[className];
   if (!cls) {
      console.log("❌ Class not found:", className);
   } else {
      const methods = cls.$methods.sort();
      methods.forEach(m => {
         try {
            const fullName = `${className} ${m}`;
            Interceptor.attach(ObjC.classes[className][m].implementation, {
               onEnter(args) {
                  console.log(`[+] Entered ${fullName}`);
               },
               onLeave(retval) {
                  console.log(`[+] Leaving ${fullName}`);
               }
            });
         } catch (err) {
            console.log(`⚠️ Could not hook ${className} ${m}:`, err);
         }
      });
   }
} else {
   console.log("❌ Objective-C runtime is not available.");
}
