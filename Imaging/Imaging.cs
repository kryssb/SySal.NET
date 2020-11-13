using System;
using SySal;
using System.Runtime.Serialization;


namespace SySal.Imaging
{
	/// <summary>
	/// Pixel format specification
	/// </summary>
    [Serializable]
	public enum PixelFormatType {GrayScale8, GrayScale16, GrayScale24, GrayScale32, PlainRGB8, InterleavedRGB8}


	/// <summary>
	/// Image format descriptor
	/// </summary>
    [Serializable]
	public struct ImageInfo
	{
		public ushort Width;
		public ushort Height;
		public ushort BitsPerPixel;
		public PixelFormatType PixelFormat; 
	}


	/// <summary>
	/// Image pixel accessor interface
	/// </summary>
	public interface IImagePixels
	{
		byte this[uint index]
		{
			get;
			set;
		}

		byte this[ushort x, ushort y, ushort channel]
		{
			get;
			set;
		}

        ushort Channels
        {
            get;
        }        
	}


	/// <summary>
	/// Image class
	/// </summary>
	public class Image
	{
		public readonly ImageInfo Info;
		public readonly IImagePixels Pixels;

		public Image(ImageInfo imginfo, IImagePixels imgpixels)
		{
			Info = imginfo;
			Pixels = imgpixels;
		}
	}

	/// <summary>
	/// Describes objective characteristics
	/// </summary>
    [Serializable]
	public struct ObjectiveSpecifications
	{
        public ushort Width;
        public ushort Height;
		public ushort WinWidth;
		public ushort WinHeight;
		public ushort OffX;
		public ushort OffY;
		public double PixelToMicronX;
		public double PixelToMicronY;
		public double RefractiveShrinkage;
	}

	/// <summary>
	/// A cluster of interesting pixels in an image
	/// </summary>
    [Serializable]
	public struct Cluster
	{
		public double X;
		public double Y;
		public uint Area;

		public struct MatrixOfInertia
		{
			public long IXX;
			public long IYY;
			public long IXY;
		}

		public MatrixOfInertia Inertia;
	}

	/// <summary>
	/// Basic colors that every frame grabber should support
	/// </summary>
    [Serializable]
	public enum ColorCode {White, Black, DarkRed, DarkGreen, DarkBlue, DarkYellow, DarkCyan, DarkMagenta, DarkGray, LightRed, LightGreen, LightBlue, LightYellow, LightCyan, LightMagenta, LightGray}


	/// <summary>
	/// RGB color code
	/// </summary>
    [Serializable]
	public struct RGBColor
	{
		public byte R;
		public byte G;
		public byte B;
	}


	/// <summary>
	/// Grabbing modes
	/// </summary>
    [Serializable]
	public enum GrabModeCode {Idle, SingleFrame, Continuous}


	/// <summary>
	/// Codes that identify standard cursors
	/// </summary>
    [Serializable]
	public enum CursorCode {Cross5, Cross9, Frame5, Frame9, Frame25}


	/// <summary>
	/// Standard zoom factors
	/// </summary>
    [Serializable]
	public enum ZoomCode {Zoom1, Zoom2, Zoom4, Zoom8, Zoom16}

    /// <summary>
    /// Transformation from pixels to micron (emulsion or stage coordinates).
    /// </summary>
    [Serializable]
    public struct Image2DToWorld
    {
        /// <summary>
        /// Identifier. A common usage is:
        /// <list type="table">
        /// <listheader><term>Part</term><description>Usage</description></listheader>
        /// <item><term>0</term><description>Brick number.</description></item>
        /// <item><term>1</term><description>Plate.</description></item>
        /// <item><term>2</term><description>High dword of Zone Id.</description></item>
        /// <item><term>3</term><description>Low dword of Zone Id.</description></item>
        /// </list>        
        /// </summary>
        /// <remarks>Parts 2 and 3 of the identifier are often set to 0.</remarks>
        public SySal.BasicTypes.Identifier Id;
        /// <summary>
        /// Center of the field of view.
        /// </summary>
        public SySal.BasicTypes.Vector2 Center;
        /// <summary>
        /// Pixel-to-micron matrix, XX component.
        /// </summary>
        public double PixelToMicronXX;
        /// <summary>
        /// Pixel-to-micron matrix, XY component.
        /// </summary>
        public double PixelToMicronXY;
        /// <summary>
        /// Pixel-to-micron matrix, YX component.
        /// </summary>
        public double PixelToMicronYX;
        /// <summary>
        /// Pixel-to-micron matrix, YY component.
        /// </summary>
        public double PixelToMicronYY;
    }

    /// <summary>
    /// Depth information of images.
    /// </summary>
    [Serializable]
    public struct ImageDepthInfo
    {
        /// <summary>
        /// The Z coordinate where this image has been taken.
        /// </summary>
        public double Z;
    }

    /// <summary>
    /// Depth information for images in an emulsion layer.
    /// </summary>
    [Serializable]
    public struct EmulsionLayerImageDepthInfo
    {
        /// <summary>
        /// Top Z of the emulsion layer.
        /// </summary>
        public double TopZ;
        /// <summary>
        /// Bottom Z of the emulsion layer.
        /// </summary>
        public double BottomZ;
        /// <summary>
        /// Depth information for each image.
        /// </summary>
        public ImageDepthInfo [] DepthInfo;
    }

    /// <summary>
    /// Information about a tomographic sequence of images.
    /// </summary>
    [Serializable]
    public struct ImageSequenceInfo
    {
        /// <summary>
        /// Transformation parameters of 2D images.
        /// </summary>
        public Image2DToWorld Info2D;
        /// <summary>
        /// Depth information for all emulsion layers.
        /// </summary>
        public EmulsionLayerImageDepthInfo[] EmulsionLayers;
        /// <summary>
        /// Comment about the image sequence.
        /// </summary>
        public string Comment;
    }
}
