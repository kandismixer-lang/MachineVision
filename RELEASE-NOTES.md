• เปลี่ยนชื่อโปรแกรมเป็น FrameLab by IRiSH พร้อมโลโก้และไอคอนใหม่ การตั้งค่าและเทมเพลตเดิมยังอยู่ครบ

• เครื่องมือประมวลผลภาพใหม่หลายสิบตัว เช่น Active Contour, Deconvolution แบบ blind/regularized, MS-SSIM, Exposure Fusion, Local Laplacian, Rank Filter, Custom LUT, Image Stats, corr2, ทิศทาง gradient, BF Score, Mask Overlay, Perimeter/Thicken, Reconstruct, Decorrstretch, Quadtree, Radon, Geodesic Distance, Test Pattern, Phase Correlation, ISODATA, ออกแบบ FIR filter และ Inpaint แบบ exemplar

• แปรงวาดเส้นแดง/น้ำเงินลงบนภาพได้ในแอป (มี undo/redo และยางลบ) ใช้กับ Scribble Segmentation แบ่งวัตถุจากเส้นที่วาด

• สีใหม่: YIQ, sRGB linear, Color Matrix, White Balance แบบ PCA และ Delta E เพิ่มสูตร CIE94

• Pass/Fail วัดค่าจากเครื่องมือใหม่ได้ เช่น พื้นที่ Active Contour, MS-SSIM และค่าสถิติภาพ

• โค้ด Python ที่ export จะเปิดภาพ/กล้อง/วิดีโอเดียวกับที่ใช้ในแอปให้เลย ไม่ต้องแก้ชื่อไฟล์ input.jpg เอง

• แก้ Edgetaper ให้ภาพไม่ดำบนภาพใหญ่และเร็วขึ้น และ Region Fill ให้ผลแม่นขึ้นบนรูขนาดใหญ่

• แก้การแปลงโมเดล .pt ที่เลือกไฟล์ไม่ได้
