using System;
using System.Collections.Generic;
using System.Text;

namespace SySal.Executables.NExTScanner
{
    /// <summary>
    /// Parameters for image handling.
    /// </summary>
    [Serializable]
    public class ImagingConfiguration : SySal.Management.Configuration
    {
        /// <summary>
        /// Width of the image.
        /// </summary>
        public uint ImageWidth = 1280;
        /// <summary>
        /// Height of the image.
        /// </summary>
        public uint ImageHeight = 1024;
        /// <summary>
        /// Empty image descriptor.
        /// </summary>
        /// <remarks>The image can be provided in the formats:
        /// <list type="table">
        /// <listheader><term>Format</term><description>Description/Example</description></listheader>
        /// <item><term>Fit</term><description>The image is obtained from an interpolation. <example>fit:2,2;100,100,5;1100,100,2;100,900,20;1100,900,5</example></description></item>
        /// <item><term>Base64</term><description>The image is gzipped, then encoded in base64 format. <example>base64:00AbxWef....</example></description></item>
        /// </list>
        /// </remarks>
        public string EmptyImage = "";
        /// <summary>
        /// Target median of the grey level for image equalization.
        /// </summary>
        public uint GreyTargetMedian = 220;
        /// <summary>
        /// Threshold image descriptor.
        /// </summary>
        /// <remarks>The image can be provided in the formats:
        /// <list type="table">
        /// <listheader><term>Format</term><description>Description/Example</description></listheader>
        /// <item><term>Fit</term><description>The image is obtained from an interpolation. <example>fit:2,2;100,100,5;1100,100,2;100,900,20;1100,900,5</example></description></item>
        /// <item><term>Base64</term><description>The image is gzipped, then encoded in base64 format. <example>base64:00AbxWef....</example></description></item>
        /// </list>
        /// </remarks>
        public string ThresholdImage;
        /// <summary>
        /// Maximum number of segments per scan line.
        /// </summary>
        public uint MaxSegmentsPerLine = 100;
        /// <summary>
        /// Maximum number of clusters per image.
        /// </summary>
        public uint MaxClusters = 30000;
        /// <summary>
        /// Pixel-micron conversion factors.
        /// </summary>
        public SySal.BasicTypes.Vector2 Pixel2Micron;
        /// <summary>
        /// X slant of the optical axis.
        /// </summary>
        public double XSlant = 0.0;
        /// <summary>
        /// Y slant of the optical axis.
        /// </summary>
        public double YSlant = 0.0;
        /// <summary>
        /// The variation of the magnification with X.
        /// </summary>
        public double DMagDX = 0.0;
        /// <summary>
        /// The variation of the magnification with Y.
        /// </summary>
        public double DMagDY = 0.0;
        /// <summary>
        /// The variation of the magnification with Z.
        /// </summary>
        public double DMagDZ = 0.0;
        /// <summary>
        /// The coefficient D<sup>2</sup>Z/DRadius<sup>2</sup> (off-plane curvature). 
        /// </summary>
        public double ZCurvature = 0.0;
        /// <summary>
        /// The coefficient DX/(D(Y<sup>2</sup>)DX) or DY/(D(X<sup>2</sup>)DY) (in-plane curvature).
        /// </summary>
        public double XYCurvature = 0.0;
        /// <summary>
        /// Rotation of the camera axis.
        /// </summary>
        public double CameraRotation = 0.0;
        /// <summary>
        /// Position tolerance to be used to put two clusters in the same grain.
        /// </summary>
        public double ClusterMatchPositionTolerance = 0.3;
        /// <summary>
        /// Maximum offset between two layers of clusters.
        /// </summary>
        public double ClusterMatchMaxOffset = 3.0;
        /// <summary>
        /// Minimum size for clusters to be eligible to form grains.
        /// </summary>        
        public uint MinClusterArea = 2;
        /// <summary>
        /// Minimum total sum of areas of all clusters in a grain.
        /// </summary>
        public uint MinGrainVolume = 4;
        /// <summary>
        /// Minimum number of matches between two layers.
        /// </summary>
        public uint MinClusterMatchCount = 100;

        public ImagingConfiguration() : base("")
        {
            SySal.Imaging.ImageInfo info = new SySal.Imaging.ImageInfo();
            info.Width = (ushort)ImageWidth;
            info.Height = (ushort)ImageHeight;
            info.PixelFormat = SySal.Imaging.PixelFormatType.GrayScale8;
            info.BitsPerPixel = 16;
            SySal.Imaging.DCTInterpolationImage.PointValue [] pval = new SySal.Imaging.DCTInterpolationImage.PointValue[1];
            pval[0].X = (ushort)(info.Width / 2);
            pval[0].Y = (ushort)(info.Height / 2);
            pval[0].Value = 200;
            ThresholdImage = new SySal.Imaging.DCTInterpolationImage(info, 1, 1, pval).ToString();
            Pixel2Micron.X = -0.3;
            Pixel2Micron.Y = 0.3;
        }

        internal static ImagingConfiguration Default
        {
            get
            {
                ImagingConfiguration c = SySal.Management.MachineSettings.GetSettings(typeof(ImagingConfiguration)) as ImagingConfiguration;
                if (c == null) c = new ImagingConfiguration();
                return c;
            }
        }

        #region ICloneable Members

        public override object Clone()
        {            
            ImagingConfiguration c = new ImagingConfiguration();
            c.Name = Name;            
            c.EmptyImage = EmptyImage;
            c.GreyTargetMedian = GreyTargetMedian;
            c.ThresholdImage = ThresholdImage;
            c.MaxSegmentsPerLine = MaxSegmentsPerLine;
            c.MaxClusters = MaxClusters;
            c.ImageWidth = ImageWidth;
            c.ImageHeight = ImageHeight;
            c.Pixel2Micron = Pixel2Micron;
            c.ZCurvature = ZCurvature;
            c.DMagDX = DMagDX;
            c.DMagDY = DMagDY;
            c.DMagDZ = DMagDZ;
            c.XYCurvature = XYCurvature;
            c.CameraRotation = CameraRotation;
            c.XSlant = XSlant;
            c.YSlant = YSlant;
            c.ClusterMatchPositionTolerance = ClusterMatchPositionTolerance;
            c.ClusterMatchMaxOffset = ClusterMatchMaxOffset;
            c.MinClusterArea = MinClusterArea;
            c.MinGrainVolume = MinGrainVolume;
            c.MinClusterMatchCount = MinClusterMatchCount;
            return c;
        }

        public void Copy(ImagingConfiguration c)
        {
            Name = c.Name;
            EmptyImage = c.EmptyImage;
            GreyTargetMedian = c.GreyTargetMedian;
            ThresholdImage = c.ThresholdImage;
            MaxSegmentsPerLine = c.MaxSegmentsPerLine;
            MaxClusters = c.MaxClusters;
            ImageWidth = c.ImageWidth;
            ImageHeight = c.ImageHeight;
            Pixel2Micron = c.Pixel2Micron;
            DMagDX = c.DMagDX;
            DMagDY = c.DMagDY;
            DMagDZ = c.DMagDZ;            
            ZCurvature = c.ZCurvature;
            XYCurvature = c.XYCurvature;
            CameraRotation = c.CameraRotation;
            XSlant = c.XSlant;
            YSlant = c.YSlant;
        }
        
        #endregion

        public const string FileExtension = "imaging.config";

        public override string ToString()
        {
            return
                "<ImageCorrection>\r\n" +
                " <XSlant>" + XSlant.ToString(System.Globalization.CultureInfo.InvariantCulture) + "</XSlant>\r\n" +
                " <YSlant>" + YSlant.ToString(System.Globalization.CultureInfo.InvariantCulture) + "</YSlant>\r\n" +
                " <DMagDX>" + DMagDX.ToString(System.Globalization.CultureInfo.InvariantCulture) + "</DMagDX>\r\n" +
                " <DMagDY>" + DMagDY.ToString(System.Globalization.CultureInfo.InvariantCulture) + "</DMagDY>\r\n" +
                " <DMagDZ>" + DMagDZ.ToString(System.Globalization.CultureInfo.InvariantCulture) + "</DMagDZ>\r\n" +
                " <XYCurvature>" + XYCurvature.ToString(System.Globalization.CultureInfo.InvariantCulture) + "</XYCurvature>\r\n" +
                " <ZCurvature>" + ZCurvature.ToString(System.Globalization.CultureInfo.InvariantCulture) + "</ZCurvature>\r\n" +
                " <CameraRotation>" + CameraRotation.ToString(System.Globalization.CultureInfo.InvariantCulture) + "</CameraRotation>\r\n" +
                "</ImageCorrection>\r\n";
        }
    }

    internal class SySalImageFromImage : SySal.Imaging.Image
    {
        public static System.Drawing.Image ThresholdImage(SySal.Imaging.DCTInterpolationImage dct)
        {
            int maxsize = Math.Max(dct.Info.Width, dct.Info.Height);
            double f = Math.Min(320, maxsize) / (double)maxsize;
            int w = (int)(f * dct.Info.Width);
            int w1 = w + 1;            
            int h = (int)(f * dct.Info.Height);
            int h1 = h + 1;
            NumericalTools.Plot pl = new NumericalTools.Plot();
            System.Drawing.Bitmap bmp = new System.Drawing.Bitmap(w * 2 + 100, h * 2 + 100);
            double[] x = new double[w1 * h1];
            double[] y = new double[w1 * h1];
            double[] t = new double[w1 * h1];
            int ix = 0, iy = 0;
                for (iy = 0; iy < h; iy++)
                    for (ix = 0; ix < w; ix++)
                    {
                        x[iy * w1 + ix] = ix / f;
                        y[iy * w1 + ix] = -iy / f;
                        t[iy * w1 + ix] = (double)((dct.Pixels[(ushort)x[iy * w1 + ix], (ushort)(-y[iy * w1 + ix]), 0] << 8) + dct.Pixels[(ushort)x[iy * w1 + ix], (ushort)(-y[iy * w1 + ix]), 1]);
                    }
            double minth = 0.0, maxth = 0.0, avgth = 0.0, rmsth = 0.0;
            NumericalTools.Fitting.FindStatistics(t, ref minth, ref maxth, ref avgth, ref rmsth);            
            pl.LabelFont = new System.Drawing.Font("Segoe UI", 12);
            pl.PanelX = 2.0;
            pl.PanelY = 0.0;
            pl.VecX = x;
            pl.VecY = y;
            pl.VecZ = t;
            pl.DX = (float)(dct.Info.Width / w);
            pl.DY = (float)(dct.Info.Height / h);
            pl.MinX = 0.0;
            pl.MinX = dct.Info.Width;
            pl.MinY = -dct.Info.Height;
            pl.MaxY = 0.0;
            pl.MinZ = minth - 10.0;
            pl.MaxZ = maxth + 10.0;
            pl.XTitle = "X Pixels";
            pl.YTitle = "-Y Pixels";
            pl.ZTitle = "Threshold";
            pl.HueAreaComputedValues(System.Drawing.Graphics.FromImage(bmp), w * 2, h * 2);
            return bmp;
        }

        private static SySal.Imaging.ImageInfo InfoFromImage(System.Drawing.Image im)
        {
            SySal.Imaging.ImageInfo info = new SySal.Imaging.ImageInfo();
            info.Width = (ushort)im.Width;
            info.Height = (ushort)im.Height;
            info.PixelFormat = SySal.Imaging.PixelFormatType.GrayScale8;
            info.BitsPerPixel = 8;
            return info;
        }

        public SySalImageFromImage(System.Drawing.Image im)
            : base(InfoFromImage(im), new SySalPixels(im)) { }

        private class SySalPixels : SySal.Imaging.IImagePixels
        {
            System.Drawing.Bitmap m_Image;

            public SySalPixels(System.Drawing.Image im)
            {
                if (im is System.Drawing.Bitmap) m_Image = im as System.Drawing.Bitmap;
                else m_Image = new System.Drawing.Bitmap(im);
            }

            #region IImagePixels Members

            public ushort Channels
            {
                get { return 1; }
            }

            public byte this[ushort x, ushort y, ushort channel]
            {
                get
                {
                    if (channel > 0) throw new Exception("Only one channel supported.");
                    System.Drawing.Color c = m_Image.GetPixel(x, y);
                    return (byte)((c.R + c.G + c.B) / 3);
                }
                set
                {
                    throw new Exception("This image is read-only.");
                }
            }

            public byte this[uint index]
            {
                get
                {
                    int y = (int)(index / m_Image.Width);
                    int x = (int)(index % m_Image.Width);
                    System.Drawing.Color c = m_Image.GetPixel(x, y);
                    return (byte)((c.R + c.G + c.B) / 3);
                }
                set
                {
                    throw new Exception("The method or operation is not implemented.");
                }
            }

            #endregion
        }

        public static System.Drawing.Image ToImage(SySal.Imaging.Image im)
        {
            if (im.Info.BitsPerPixel != 8 && im.Info.BitsPerPixel != 16) throw new Exception("Only 8 and 16 bit images are supported.");
            bool is16bit = (im.Info.BitsPerPixel == 16);
            System.Drawing.Bitmap bmp = new System.Drawing.Bitmap(im.Info.Width, im.Info.Height, is16bit ? System.Drawing.Imaging.PixelFormat.Format16bppGrayScale : System.Drawing.Imaging.PixelFormat.Format24bppRgb);
            ushort x, y;
            if (is16bit)
                for (y = 0; y < im.Info.Height; y++)
                    for (x = 0; x < im.Info.Width; x++)
                        bmp.SetPixel(x, y, System.Drawing.Color.FromArgb((im.Pixels[x, y, (ushort)0] << 8) + im.Pixels[x, y, (ushort)1]));
            else
                for (y = 0; y < im.Info.Height; y++)
                    for (x = 0; x < im.Info.Width; x++)
                        bmp.SetPixel(x, y, System.Drawing.Color.FromArgb(im.Pixels[x, y, (ushort)0], im.Pixels[x, y, (ushort)0], im.Pixels[x, y, (ushort)0]));
            return bmp;
        }
    }
}
