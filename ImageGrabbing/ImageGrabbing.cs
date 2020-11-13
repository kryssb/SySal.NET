using System;
using System.Collections.Generic;
using System.Text;

namespace SySal.Imaging
{
    /// <summary>
    /// Defines generic methods and properties of an image grabber.
    /// </summary>
    public interface IImageGrabber : IDisposable
    {
        /// <summary>
        /// Gets the format of the images.
        /// </summary>
        ImageInfo ImageFormat { get; }
        /// <summary>
        /// The number of images grabbed in each sequence.
        /// </summary>        
        int SequenceSize
        {
            get;
            set;
        }
        /// <summary>
        /// The number of grab sequences that can be held in memory.
        /// </summary>
        int Sequences
        {
            get;
        }
        /// <summary>
        /// The number of image sequences that can be simultaneously mapped in memory.
        /// </summary>
        int MappedSequences
        {
            get;
        }
        /// <summary>
        /// Tests whether the grabber is ready to work (additional configuration may be needed depending on the implementation).
        /// </summary>
        bool IsReady
        {
            get;
        }
        /// <summary>
        /// Sets the time source.
        /// </summary>
        System.Diagnostics.Stopwatch TimeSource { set; }
        /// <summary>
        /// Grabs a set of images.
        /// </summary>        
        /// <returns>an opaque object that contains information about the grabbed data. The usage is device-specific.</returns>
        /// <remarks>The internal buffer must be cleared by calling <seealso cref="ClearGrabSequence"/></remarks>
        object GrabSequence();
        /// <summary>
        /// Gets the time stamps of the images in a sequence.
        /// </summary>
        /// <param name="gbsq">the sequence.</param>
        /// <returns>the acquisition times of the images in the buffer, in ms.</returns>
        double[] GetImageTimesMS(object gbsq);
        /// <summary>
        /// Clears a grabbed sequence. This method must be called to free the resources associated to each sequence.
        /// </summary>
        /// <param name="gbsq">the sequence to be cleared.</param>
        void ClearGrabSequence(object gbsq);
        /// <summary>
        /// Maps a sequence a single image containing all images one after the other.
        /// </summary>
        /// <param name="gbsq">the sequence to be mapped.</param>
        /// <returns>a linear memory image that contains all the images stored one after the other in Y order.</returns>
        Image MapSequenceToSingleImage(object gbsq);
        /// <summary>
        /// Clears a mapped image.
        /// </summary>
        /// <param name="img">the image to be cleared.</param>
        void ClearMappedImage(Image img);
    }

    /// <summary>
    /// Class for shape parameters computation.
    /// </summary>
    public class ClusterShape
    {
        public static void EllipseParametersVector(ref Cluster c, out double axismax, out double axismin, out SySal.BasicTypes.Vector2 axismaxv)
        {
            double delta = c.Inertia.IXX - c.Inertia.IYY;
            double discQ = Math.Sqrt(delta * delta + 4.0 * c.Inertia.IXY * c.Inertia.IXY);
            double dQ = c.Inertia.IXX + c.Inertia.IYY;
            double lambda1 = 0.5 * (dQ + discQ);
            axismax = 4.0 * Math.Sqrt(lambda1 / c.Area);
            axismin = 4.0 * Math.Sqrt(0.5 * (dQ - discQ) / c.Area);
            axismaxv.Y = c.Inertia.IXX - lambda1;
            axismaxv.X = -c.Inertia.IXY;
            double axnorm = 1.0 / Math.Sqrt(axismaxv.X * axismaxv.X + axismaxv.Y * axismaxv.Y);
            axismaxv.X += axnorm;
            axismaxv.Y += axnorm;
        }
        public static void EllipseParametersDegree(ref Cluster c, out double axismax, out double axismin, out double orientation)
        {
            double delta = c.Inertia.IXX - c.Inertia.IYY;
            double discQ = Math.Sqrt(delta * delta + 4.0 * c.Inertia.IXY * c.Inertia.IXY);
            double dQ = c.Inertia.IXX + c.Inertia.IYY;
            double lambda1 = 0.5 * (dQ + discQ);
            axismax = 4.0 * Math.Sqrt(lambda1 / c.Area);
            axismin = 4.0 * Math.Sqrt(0.5 * (dQ - discQ) / c.Area);
            orientation = Math.Atan2(c.Inertia.IXX - lambda1, -c.Inertia.IXY) * 180.0 / Math.PI;
        }
    }
    /// <summary>
    /// Features required in image processing.
    /// </summary>
    [Flags]
    public enum ImageProcessingFeatures
    {
        /// <summary>
        /// No feature.
        /// </summary>
        None = 0,
        /// <summary>
        /// Image after grey level equalization.
        /// </summary>
        EqualizedImage = 0x1,
        /// <summary>
        /// Image after filtering.
        /// </summary>
        FilteredImage = 0x2,
        /// <summary>
        /// Image after binarization.
        /// </summary>
        BinarizedImage = 0x4,
        /// <summary>
        /// The list of segments in each row.
        /// </summary>
        Segments = 0x8,
        /// <summary>
        /// The list of clusters with area and X/Y position.
        /// </summary>
        Clusters = 0x10,
        /// <summary>
        /// Second momenta (matrix of inertia) of clusters. Implies <see cref="Clusters"/>.
        /// </summary>
        Cluster2ndMomenta = 0x30
    }
    /// <summary>
    /// Segment of a cluster.
    /// </summary>
    public struct ClusterSegment
    {
        /// <summary>
        /// Scan line where the segment is found.
        /// </summary>
        public uint Line;
        /// <summary>
        /// Left pixel.
        /// </summary>
        public ushort Left;
        /// <summary>
        /// Right pixel.
        /// </summary>
        public ushort Right;
        /// <summary>
        /// The cluster that owns this segment; 
        /// </summary>
        public int Owner;
    }

    /// <summary>
    /// Defines the interesting region of an image.
    /// </summary>
    public struct ImageWindow
    {
        /// <summary>
        /// X coordinate of the left edge (included) of the window.
        /// </summary>
        public uint MinX;
        /// <summary>
        /// X coordinate of the right edge (included) of the window.
        /// </summary>
        public uint MaxX;
        /// <summary>
        /// Y coordinate of the top edge (included) of the window.
        /// </summary>
        public uint MinY;
        /// <summary>
        /// Y coordinate of the bottom edge (included) of the window.
        /// </summary>
        public uint MaxY;
    }

    /// <summary>
    /// A sequence of images.
    /// </summary>
    public interface IMultiImage
    {
        /// <summary>
        /// The number of images contained in the sequence.
        /// </summary>
        uint SubImages { get; }
        /// <summary>
        /// Retrieves the i-th image in the sequence.
        /// </summary>
        /// <param name="i">the number of the image to access.</param>
        /// <returns>the i-th sub-image.</returns>
        Image SubImage(uint i);
    }

    /// <summary>
    /// Provides encoding in base64 format.
    /// </summary>
    /// <remarks>Base64 encoded images have the following format:<br />
    /// <c>base64:width,height,bitsperpixel<br />
    /// <i>dataline0...</i><br/>
    /// <i>dataline1...</i><br/>
    /// <i>dataline2...</i><br/>
    /// ...</c>
    /// </remarks>
    public class Base64ImageEncoding
    {
        const string Base64String = "base64:";

        /// <summary>
        /// Encodes an image in Base64 format.
        /// </summary>
        /// <param name="im">the image to be encoded.</param>
        /// <returns>the string with the encoded image.</returns>
        public static string ImageToBase64(SySal.Imaging.Image im)
        {
            if (im.Info.PixelFormat != PixelFormatType.GrayScale8) throw new Exception("Unsupported format. The only supported format is " + PixelFormatType.GrayScale8 + ".");
            int bitsperpixel = (im.Info.BitsPerPixel / 8);
            byte[] b = new byte[bitsperpixel * im.Info.Width * im.Info.Height];
            int i;
            for (i = 0; i < b.Length; i++)
                b[i] = im.Pixels[(uint)i];
            System.IO.MemoryStream ms = new System.IO.MemoryStream();
            System.IO.Compression.GZipStream gz = new System.IO.Compression.GZipStream(ms, System.IO.Compression.CompressionMode.Compress);
            gz.Write(b, 0, b.Length);
            gz.Flush();
            gz.Close();
            return Base64String + im.Info.Width + "," + im.Info.Height + "," + im.Info.BitsPerPixel + "\r\n" + System.Convert.ToBase64String(ms.ToArray(), Base64FormattingOptions.InsertLineBreaks);
        }

        /// <summary>
        /// Builds an image from its Base64 encoding.
        /// </summary>
        /// <param name="encstr">the string containing the encoded image.</param>
        /// <returns>the image.</returns>
        public static SySal.Imaging.Image ImageFromBase64(string encstr)
        {
            int hpos = encstr.IndexOfAny(new char[] { '\r', '\n' });
            if (hpos < 0) throw new Exception("Can't find base64 header.");
            string hstr = encstr.Substring(0, hpos);
            if (hstr.ToLower().StartsWith(Base64String) == false) throw new Exception("Can't find header string (\"" + Base64String + "\").");
            string[] htokens = hstr.Substring(Base64String.Length).Split(',');
            if (htokens.Length != 3) throw new Exception("Header syntax is " + Base64String + ",width,height,bitsperpixel");
            SySal.Imaging.ImageInfo info = new ImageInfo();
            info.Width = ushort.Parse(htokens[0]);
            info.Height = ushort.Parse(htokens[1]);
            info.BitsPerPixel = ushort.Parse(htokens[2]);
            info.PixelFormat = PixelFormatType.GrayScale8;
            System.IO.MemoryStream ms = new System.IO.MemoryStream(System.Convert.FromBase64String(encstr.Substring(hpos)));
            byte[] b = new byte[info.Width * info.Height * (info.BitsPerPixel / 8)];
            System.IO.Compression.GZipStream gz = new System.IO.Compression.GZipStream(ms, System.IO.Compression.CompressionMode.Decompress);
            gz.Read(b, 0, b.Length);
            return new SySal.Imaging.Image(info, new ImagePixels(info, b));
        }

        internal class ImagePixels : SySal.Imaging.IImagePixels
        {
            SySal.Imaging.ImageInfo m_Info;
            byte[] m_Bytes;

            internal ImagePixels(SySal.Imaging.ImageInfo info, byte[] b)
            {
                m_Info = info;
                m_Bytes = b;
            }

            #region IImagePixels Members

            public byte this[uint index]
            {
                get
                {
                    return m_Bytes[index];
                }
                set
                {
                    throw new Exception("This image is read-only.");
                }
            }

            public byte this[ushort x, ushort y, ushort channel]
            {
                get
                {
                    return m_Bytes[(y * m_Info.Width + x) * (m_Info.BitsPerPixel / 8) + channel];
                }
                set
                {
                    throw new Exception("This image is read-only.");
                }
            }

            public ushort Channels
            {
                get { return (ushort)(m_Info.BitsPerPixel / 8); }
            }

            #endregion
        }
    }

    /// <summary>
    /// Image obtained by interpolating the values provided by several points using a Discrete Cosine Transform.
    /// </summary>
    public class DCTInterpolationImage : SySal.Imaging.Image
    {
        /// <summary>
        /// Value of an image in a point.
        /// </summary>
        public struct PointValue
        {
            /// <summary>
            /// X coordinate.
            /// </summary>
            public ushort X;
            /// <summary>
            /// Y coordinate.
            /// </summary>
            public ushort Y;
            /// <summary>
            /// Value of the image pixel.
            /// </summary>
            public int Value;
        }

        private class Zero : NumericalTools.Minimization.ITargetFunction
        {
            int m_ParamCount;

            public Zero(int parcount)
            {
                m_ParamCount = parcount;
            }

            #region ITargetFunction Members

            public int CountParams
            {
                get { return m_ParamCount; }
            }

            public NumericalTools.Minimization.ITargetFunction Derive(int i)
            {
                return this;
            }

            public double Evaluate(params double[] x)
            {
                return 0.0;
            }

            public double RangeMax(int i)
            {
                return 1e9;
            }

            public double RangeMin(int i)
            {
                return -1e9;
            }

            public double[] Start
            {
                get { throw new Exception("The method or operation is not implemented."); }
            }

            public bool StopMinimization(double fval, double fchange, double xchange)
            {
                throw new Exception("The method or operation is not implemented.");
            }

            #endregion
        }

        /// <summary>
        /// Discrete Cosine Transform
        /// </summary>
        protected class DCT : NumericalTools.Minimization.ITargetFunction, SySal.Imaging.IImagePixels
        {
            private int XWaves, YWaves;
            private PointValue[] PVals;
            private int XMax, YMax;
            private double IXMax, IYMax;
            internal double[] ParamValues = null;
            private int Chnls;
            private int IDeriv;

            public DCT(int channels, int xw, int yw, int xm, int ym, PointValue[] pvals)
            {
                Chnls = channels;
                XWaves = xw;
                YWaves = yw;
                XMax = xm;
                YMax = ym;
                IXMax = 1.0 / XMax;
                IYMax = 1.0 / YMax;
                PVals = pvals;
                IDeriv = -1;
            }

            public DCT(int channels, int xw, int yw, int xm, int ym, PointValue[] pvals, int ideriv)
            {
                Chnls = channels;
                XWaves = xw;
                YWaves = yw;
                XMax = xm;
                YMax = ym;
                IXMax = 1.0 / XMax;
                IYMax = 1.0 / YMax;
                PVals = pvals;
                IDeriv = ideriv;
            }

            #region ITargetFunction Members

            public int CountParams
            {
                get { return 2 + XWaves * YWaves; }
            }

            public NumericalTools.Minimization.ITargetFunction Derive(int i)
            {
                return (IDeriv >= 0) ? (NumericalTools.Minimization.ITargetFunction)new Zero(2 + XWaves * YWaves) : (NumericalTools.Minimization.ITargetFunction)new DCT(Channels, XWaves, YWaves, XMax, YMax, PVals, i);
            }

            internal double[,] XCosValues;
            internal double[,] YCosValues;

            internal void MakeCosValues()
            {
                int ix = 0, iy = 0, i = 0;
                XCosValues = new double[XWaves, XMax];
                YCosValues = new double[YWaves, YMax];
                double invx = Math.PI / XMax;
                double invy = Math.PI / YMax;
                for (ix = 0; ix < XWaves; ix++)
                    for (i = 0; i < XMax; i++)
                        XCosValues[ix, i] = Math.Cos(ix * i * invx);

                for (iy = 0; iy < YWaves; iy++)
                    for (i = 0; i < YMax; i++)
                        YCosValues[iy, i] = Math.Cos(iy * i * invy);
            }

            private double XYEval(double x, double y, double[] p)
            {
                if (IDeriv >= 0) return Math.Cos(x * (IDeriv % XWaves)) * Math.Cos(y * (IDeriv / XWaves));
                double s = 0.0;
                int ix, iy;
                for (ix = 0; ix < XWaves; ix++)
                    for (iy = 0; iy < YWaves; iy++)
                        s += p[iy * XWaves + ix] * Math.Cos(x * ix) * Math.Cos(y * iy);
                return s;
            }

            public double Evaluate(params double[] p)
            {
                double[] q = new double[p.Length - 2];
                int i;
                for (i = 0; i < p.Length - 2; i++) q[i] = p[i];
                return XYEval(p[p.Length - 2] * IXMax * Math.PI, p[p.Length - 1] * IYMax * Math.PI, q);
            }

            public double RangeMax(int i)
            {
                return 32767;
            }

            public double RangeMin(int i)
            {
                return -32767;
            }

            public double[] Start
            {
                get { return new double[XWaves * YWaves]; }
            }

            public bool StopMinimization(double fval, double fchange, double xchange)
            {
                return fval < 1.0e-3;
            }

            #endregion

            #region IImagePixels Members

            public ushort Channels
            {
                get { return (ushort)Chnls; }
            }

            internal int[] m_CachedValues;

            public byte this[ushort x, ushort y, ushort channel]
            {
                get
                {
                    int ch = Chnls - 1 - channel;
                    if (m_CachedValues != null)                    
                        return (byte)(((m_CachedValues[y * XMax + x]) & (0xFF << (8 * ch))) >> (8 * ch));                    
                    //int v = (int)Math.Round(XYEval(x * IXMax * Math.PI, y * IYMax * Math.PI, ParamValues));
                    int ix, iy;
                    double dv = 0.0;
                    for (ix = 0; ix < XWaves; ix++)
                        for (iy = 0; iy < YWaves; iy++)
                            dv += ParamValues[iy * XWaves + ix] * XCosValues[ix, x] * YCosValues[iy, y];
                    int v = (int)Math.Round(dv);
                    return (byte)((v & (0xFF << (8 * ch))) >> (8 * ch));
                }
                set
                {
                    throw new Exception("The method or operation is not implemented.");
                }
            }

            public byte this[uint index]
            {
                get
                {
                    int i = (int)(index / Chnls);
                    int channel = Chnls - 1 - (int)(index % Chnls);
                    //int v = (int)Math.Round(XYEval((i % XMax) * IXMax * Math.PI, (i / XMax) * IYMax * Math.PI, ParamValues));
                    if (m_CachedValues != null)
                        return (byte)(((m_CachedValues[i]) & (0xFF << (8 * channel))) >> (8 * channel));                    
                    int ix, iy;
                    double dv = 0.0;
                    for (ix = 0; ix < XWaves; ix++)
                        for (iy = 0; iy < YWaves; iy++)
                            dv += ParamValues[iy * XWaves + ix] * XCosValues[ix, i % XMax] * YCosValues[iy, i / XMax];
                    int v = (int)Math.Round(dv);
                    return (byte)((v & (0xFF << (8 * channel))) >> (8 * channel));
                }
                set
                {
                    throw new Exception("The method or operation is not implemented.");
                }
            }

            #endregion
        }

        protected int XWaves, YWaves;

        protected PointValue[] PointValues;

        /// <summary>
        /// Produces a string representation of this image.
        /// </summary>
        /// <returns>a string representation of the image.</returns>
        public override string ToString()
        {
            string t = "";
            foreach (PointValue p in PointValues)
                t += ";" + p.X + "," + p.Y + "," + p.Value;
            return "dct:" + Info.Width + "," + Info.Height + "," + Info.BitsPerPixel + "," + XWaves + "," + YWaves + t;
        }

        const string DCTString = "dct:";

        /// <summary>
        /// Builds a DCT-interpolated image from its string representation.
        /// </summary>
        /// <param name="fitstring">the string containing the representation of the image.</param>
        /// <remarks>
        /// Syntax for string representation of DCT-coded images:<br />
        /// <c>dct:<i>width</i>,<i>height</i>,<i>bitsperpixel</i>,<i>xwaves</i>,<i>ywaves</i>{;<i>x</i>,<i>y</i>,<i>value</i>}</c><br />
        /// <example>dct:1280,1024,8,2,2;100,100,180;1000,120,220;1004,900,200;180,940,200</example>
        /// </remarks>
        public static DCTInterpolationImage FromDCTString(string fitstring)
        {
            lock (s_LockObj)
                if (s_LastImageText != null && fitstring == s_LastImageText) return s_LastImage;
            if (fitstring.ToLower().StartsWith(DCTString) == false) throw new Exception("DCT coding string must begin with \"dct:\".");
            string[] tokens = fitstring.Substring(DCTString.Length).Split(';');
            string[] htokens = tokens[0].Split(',');
            if (htokens.Length != 5) throw new Exception("DCT string must contain width, height, bitsperpixel, xwaves, ywaves.");
            SySal.Imaging.ImageInfo info = new ImageInfo();
            info.Width = ushort.Parse(htokens[0]);
            info.Height = ushort.Parse(htokens[1]);
            info.BitsPerPixel = ushort.Parse(htokens[2]);
            info.PixelFormat = PixelFormatType.GrayScale8;
            int xwaves = int.Parse(htokens[3]);
            int ywaves = int.Parse(htokens[4]);
            int i;
            PointValue [] pvals = new PointValue[tokens.Length - 1];
            for (i = 0; i < pvals.Length; i++)
            {
                htokens = tokens[i + 1].Split(',');
                if (htokens.Length != 3) throw new Exception("Error in point value specification " + i + ": point value syntax is x,y,value.");
                pvals[i].X = ushort.Parse(htokens[0]);
                pvals[i].Y = ushort.Parse(htokens[1]);
                pvals[i].Value = int.Parse(htokens[2]);
            }
            lock (s_LockObj)
            {
                s_LastImageText = fitstring;
                s_LastImage = new DCTInterpolationImage(info, xwaves, ywaves, pvals); 
                return s_LastImage;
            }
        }

        static DCTInterpolationImage s_LastImage = null;
        static string s_LastImageText = null;
        static object s_LockObj = new object();

        /// <summary>
        /// Builds a DCT-interpolated image.
        /// </summary>
        /// <param name="info">format of the image.</param>
        /// <param name="xwaves">number of X waves.</param>
        /// <param name="ywaves">number of Y waves.</param>
        /// <param name="pval">the values of a set of sampled points.</param>
        public DCTInterpolationImage(SySal.Imaging.ImageInfo info, int xwaves, int ywaves, PointValue[] pval)
            : base(info, new DCT(info.BitsPerPixel / 8, xwaves, ywaves, info.Width, info.Height, pval))
        {
            XWaves = xwaves;
            YWaves = ywaves;
            PointValues = pval;
            NumericalTools.AdvancedFitting.LeastSquares lsq = new NumericalTools.AdvancedFitting.LeastSquares();
            NumericalTools.Minimization.ITargetFunction dct = (NumericalTools.Minimization.ITargetFunction)Pixels;
            double[][] indep = new double[pval.Length][];
            double[] dep = new double[pval.Length];
            double[] deperr = new double[pval.Length];
            int i;
            for (i = 0; i < indep.Length; i++)
            {
                indep[i] = new double[2] { pval[i].X, pval[i].Y };
                dep[i] = pval[i].Value;
                deperr[i] = 1.0;
            }
            double[] p = lsq.Fit(dct, xwaves * ywaves, indep, dep, deperr, 10);
            ((DCT)dct).ParamValues = p;
            ((DCT)dct).MakeCosValues();
            ((DCT)dct).m_CachedValues = new int[info.Width * info.Height];
            System.Threading.Thread[] hcomp_Threads = new System.Threading.Thread[info.Height];
            int iiy;
            for (iiy = 0; iiy < info.Height; iiy++)
            {
                hcomp_Threads[iiy] = new System.Threading.Thread(new System.Threading.ParameterizedThreadStart(delegate(object oiiiy)
                    {
                        int iiiy = (int)oiiiy;
                        int iix, ix, iy;
                        for (iix = 0; iix < info.Width; iix++)
                        {
                            double dv = 0.0;
                            for (ix = 0; ix < XWaves; ix++)
                                for (iy = 0; iy < YWaves; iy++)
                                    dv += ((DCT)dct).ParamValues[iy * XWaves + ix] * ((DCT)dct).XCosValues[ix, iix] * ((DCT)dct).YCosValues[iy, iiiy];
                            ((DCT)dct).m_CachedValues[iiiy * info.Width + iix] = (int)Math.Round(dv);
                        }
                    }));
                hcomp_Threads[iiy].Start(iiy);
            }
            for (iiy = 0; iiy < info.Height; iiy++)
            {
                hcomp_Threads[iiy].Join();
                hcomp_Threads[iiy] = null;
            }
        }
    }

    /// <summary>
    /// An image stored in a linear buffer (DMA-usable locked pages).
    /// </summary>
    public abstract class LinearMemoryImage : Image, IMultiImage, IDisposable
    {
        /// <summary>
        /// Points to the memory address.
        /// </summary>
        protected IntPtr m_MemoryAddress;
        /// <summary>
        /// Grants access to the memory address of a LinearMemoryImage to another LinearMemoryImage
        /// </summary>
        /// <param name="im">the image to access.</param>
        /// <returns>the memory address of pixels.</returns>
        protected static IntPtr AccessMemoryAddress(LinearMemoryImage im) { return im.m_MemoryAddress; }
        /// <summary>
        /// Protected constructor. Prevents creation of a LinearMemoryImage, unless through derivation.
        /// </summary>
        protected LinearMemoryImage(ImageInfo info, IntPtr memaddr, uint subimgs, IImagePixels pixels) : base(info, pixels) { m_MemoryAddress = memaddr; m_SubImages = subimgs; }
        /// <summary>
        /// Stores the number of images contained.
        /// </summary>
        protected uint m_SubImages;

        #region IMultiImage Members
        /// <summary>
        /// The number of images contained.
        /// </summary>
        public virtual uint SubImages
        {
            get { return m_SubImages; }
        }
        /// <summary>
        /// Retrieves the i-th sub-image.
        /// </summary>
        /// <param name="i">the index of the image sought.</param>
        /// <returns>the i-th sub-image</returns>
        public abstract Image SubImage(uint i);

        #endregion

        #region IDisposable Members

        /// <summary>
        /// Deletes the object and releases all associated resources.
        /// </summary>
        public abstract void Dispose();

        #endregion
    }

    /// <summary>
    /// Exception arising during image processing.
    /// </summary>
    public class ImageProcessingException : Exception
    {
        /// <summary>
        /// Index of the sub-image that caused the exception.
        /// </summary>
        public uint SubImageIndex;
        /// <summary>
        /// Builds a new exception.
        /// </summary>
        /// <param name="text">the text of the error message.</param>
        public ImageProcessingException(uint idx, string text) : base(text) { SubImageIndex = idx; }
        /// <summary>
        /// Builds a generic image exception.
        /// </summary>
        public ImageProcessingException() : base("Unspecified Image Processing Exception") { SubImageIndex = 0; }
        /// <summary>
        /// Casts the exception as a string of text.
        /// </summary>
        /// <returns>the string representation of the exception.</returns>
        public override string ToString()
        {
            return "SubImage #" + SubImageIndex + ": " + base.ToString();
        }
        /// <summary>
        /// Returns an error message.
        /// </summary>
        public override string Message
        {
            get
            {
                return "SubImage #" + SubImageIndex + ": " + base.Message;
            }
        }
    }

    /// <summary>
    /// Interface for image processing, from raw data to clusters.
    /// </summary>
    public interface IImageProcessor : IDisposable
    {
        /// <summary>
        /// Format of the images to be processed.
        /// </summary>
        ImageInfo ImageFormat
        {
            get;
            set;
        }

        /// <summary>
        /// Tests whether the processor is ready to work (additional configuration may be needed depending on the implementation).
        /// </summary>
        bool IsReady
        {
            get;
        }

        /// <summary>
        /// Sets an empty image to be used as a reference.
        /// </summary>
        Image EmptyImage
        {
            set;
        }

        /// <summary>
        /// Sets a threshold image.
        /// </summary>
        Image ThresholdImage
        {
            set;
        }

        /// <summary>
        /// Maximum number of segments per scan line.
        /// </summary>
        /// <remarks>Some algorithms may ignore this setting.</remarks>
        uint MaxSegmentsPerScanLine
        {
            set;
        }

        /// <summary>
        /// Maximum number of clusters per image.
        /// </summary>
        /// <remarks>Some algorithms may ignore this setting.</remarks>
        uint MaxClustersPerImage
        {
            set;
        }

        /// <summary>
        /// Reference value for the median of the grey level histogram.
        /// </summary>
        /// <remarks>The grey level histogram is distorted so that its median matches this value.</remarks>
        byte EqGreyLevelTargetMedian
        {
            set;
        }

        /// <summary>
        /// Maximum number of images that can be processed in a single run.
        /// </summary>
        int MaxImages
        {
            get;
        }

        /// <summary>
        /// Median of the grey level histogram.
        /// </summary>
        uint GreyLevelMedian
        {
            get;
        }

        /// <summary>
        /// Required output data. More output requires longer times, so reducing the types of information speeds up.
        /// </summary>
        ImageProcessingFeatures OutputFeatures
        {
            get;
            set;
        }

        /// <summary>
        /// The region of the image that should be used for processing.
        /// </summary>
        ImageWindow ProcessingWindow
        {
            get;
            set;
        }

        /// <summary>
        /// Set this property to start processing on a set of images.
        /// </summary>
        LinearMemoryImage Input
        {
            set;
        }

        /// <summary>
        /// Read this property to get the set of equalized images. It is set to <c>null</c> if this output has not been requested.
        /// </summary>
        LinearMemoryImage EqualizedImages
        {
            get;
        }

        /// <summary>
        /// Read this property to get the set of filtered images. It is set to <c>null</c> if this output has not been requested.
        /// </summary>
        LinearMemoryImage FilteredImages
        {
            get;
        }

        /// <summary>
        /// Read this property to get the set of binarized images. It is set to <c>null</c> if this output has not been requested.
        /// </summary>
        LinearMemoryImage BinarizedImages
        {
            get;
        }

        /// <summary>
        /// Read this property to get the set of segments. It is set to <c>null</c> if this output has not been requested.
        /// </summary>
        /// <remarks>The output is an array of segment arrays; each segment array reports the segments found in the corresponding image.</remarks>
        ClusterSegment[][] ClusterSegments
        {
            get;
        }
        /// <summary>
        /// Read this property to get the set of clusters. It is set to <c>null</c> if this output has not been requested.
        /// </summary>
        /// <remarks>The output is an array of cluster arrays; each cluster array reports the segments found in the corresponding image.</remarks>
        Cluster[][] Clusters
        {
            get;
        }
        /// <summary>
        /// This field is set by the last processing data set. Non-fatal errors that occur during processing are reported here.
        /// </summary>
        ImageProcessingException[] Warnings
        {
            get;
        }
        /// <summary>
        /// Creates a LinearMemoryImage from a file.
        /// </summary>
        /// <param name="filename">the name of the file that contains the image.</param>
        /// <returns>the LinearMemoryImage that has been loaded.</returns>
        LinearMemoryImage ImageFromFile(string filename);
        /// <summary>
        /// Saves a LinearMemoryImage to a file.
        /// </summary>        
        /// <param name="im">the LinearMemoryImage to be saved.</param>
        /// <param name="filename">the name of the file that will host the image.</param>
        void ImageToFile(LinearMemoryImage im, string filename);
    }

    /// <summary>
    /// Extends the functions of IImageGrabber to allow scheduling grabbing at a certain time.
    /// </summary>
    public interface IImageGrabberWithTimer
    {
        /// <summary>
        /// Specifies that no urgent grabbing task is required, so internal management can occur now.
        /// </summary>
        void Idle();
        /// <summary>
        /// Grabs a set of images at a specified time.
        /// </summary>
        /// <param name="timems">the time in ms when grabbing has to start.</param>
        /// <returns>an opaque object that contains information about the grabbed data. The usage is device-specific.</returns>
        /// <remarks>The internal buffer must be cleared by calling <seealso cref="ClearGrabSequence"/></remarks>
        object GrabSequenceAtTime(long timems);

    }
}
