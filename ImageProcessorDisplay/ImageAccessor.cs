using System;
using System.Collections.Generic;
using System.Text;

namespace SySal.ImageProcessorDisplay
{
    class ImageAccessor : SySal.Imaging.LinearMemoryImage
    {
        static public IntPtr Scan(SySal.Imaging.LinearMemoryImage lm)
        {
            return SySal.Imaging.LinearMemoryImage.AccessMemoryAddress(lm);
        }

        public ImageAccessor() : base(new SySal.Imaging.ImageInfo(), new IntPtr(), 0, null) { }

        public override SySal.Imaging.Image SubImage(uint i)
        {
            return null;
        }

        public override void Dispose()
        {            
        }

    }
}
