using System;
using System.Collections;
using System.Drawing;
using System.Xml.Serialization;
using System.Security;
[assembly:AllowPartiallyTrustedCallers]

namespace GDI3D
{
    /// <summary>
    /// A generic graphical object.
    /// </summary>
    [Serializable]
    public class GDI3DObject
    {
        /// <summary>
        /// Label for the graphical object.
        /// </summary>
        public string Label;
        /// <summary>
        /// <c>true</c> if the label is to be shown, <c>false</c> otherwise.
        /// </summary>
        public bool EnableLabel;
        /// <summary>
        /// <c>true</c> if the object is to be highlighted, <c>false</c> otherwise;
        /// </summary>
        public bool Highlight;
    }

	/// <summary>
	/// A point.
	/// </summary>
	[Serializable]
	public class Point : GDI3DObject
	{
		/// <summary>
		/// X component.
		/// </summary>
		public double X;
		/// <summary>
		/// Y component.
		/// </summary>
		public double Y;
		/// <summary>
		/// Z component.
		/// </summary>
		public double Z;
		/// <summary>
		/// Red component.
		/// </summary>
		public int R;
		/// <summary>
		/// Green component.
		/// </summary>
		public int G;
		/// <summary>
		/// Blue component.
		/// </summary>
		public int B;
		/// <summary>
		/// Owner index.
		/// </summary>
		public int Owner;
		/// <summary>
		/// Default Red component.
		/// </summary>
		public static int DefaultR = 255;
		/// <summary>
		/// Default Green component.
		/// </summary>
		public static int DefaultG = 255;
		/// <summary>
		/// Default Blue component.
		/// </summary>
		public static int DefaultB = 255;
		/// <summary>
		/// Constructs a new white point.
		/// </summary>
		/// <param name="x">X component.</param>
		/// <param name="y">Y component.</param>
		/// <param name="z">Z component.</param>
		/// <param name="owner">Owner index.</param>
		public Point(double x, double y, double z, int owner)
		{
			X = x;
			Y = y;
			Z = z;
			Owner = owner;
			R = DefaultR;
			G = DefaultG;
			B = DefaultB;
		}
		/// <summary>
		/// Constructs a colored point.
		/// </summary>
		/// <param name="x">X component.</param>
		/// <param name="y">Y component.</param>
		/// <param name="z">Z component.</param>
		/// <param name="owner">Owner index.</param>
		/// <param name="r">Red component.</param>
		/// <param name="g">Green component.</param>
		/// <param name="b">Blue component.</param>
		public Point(double x, double y, double z, int owner, int r, int g, int b)
		{
			X = x;
			Y = y;
			Z = z;
			Owner = owner;
			R = r;
			G = g;
			B = b;
		}
		/// <summary>
		/// Empty constructor.
		/// </summary>
		public Point() {}
	}
	/// <summary>
	/// A line.
	/// </summary>
	[Serializable]
	public class Line : GDI3DObject
	{
        /// <summary>
        /// <c>true</c> for dashed lines, <c>false</c> otherwise;
        /// </summary>
        public bool Dashed;
		/// <summary>
		/// First point X component;
		/// </summary>
		public double XF;
		/// <summary>
		/// First point Y component;
		/// </summary>
		public double YF;
		/// <summary>
		/// First point Z component;
		/// </summary>
		public double ZF;
		/// <summary>
		/// Second point X component;
		/// </summary>
		public double XS;
		/// <summary>
		/// Second point Y component;
		/// </summary>
		public double YS;
		/// <summary>
		/// Second point Z component;
		/// </summary>
		public double ZS;
		/// <summary>
		/// Owner index.
		/// </summary>
		public int Owner;
		/// <summary>
		/// Red component;
		/// </summary>
		public int R;
		/// <summary>
		/// Green component;
		/// </summary>
		public int G;
		/// <summary>
		/// Blue component;
		/// </summary>
		public int B;
		/// <summary>
		/// Default Red component.
		/// </summary>
		public static int DefaultR = 255;
		/// <summary>
		/// Default Green component.
		/// </summary>
		public static int DefaultG = 255;
		/// <summary>
		/// Default Blue component.
		/// </summary>
		public static int DefaultB = 255;
		/// <summary>
		/// Constructs a line with default color.
		/// </summary>
		/// <param name="xf">X component of first point.</param>
		/// <param name="yf">Y component of first point.</param>
		/// <param name="zf">Z component of first point.</param>
		/// <param name="xs">X component of second point.</param>
		/// <param name="ys">Y component of second point.</param>
		/// <param name="zs">Z component of second point.</param>
		/// <param name="owner">Owner index.</param>
		public Line(double xf, double yf, double zf, double xs, double ys, double zs, int owner)
		{
			XS = xs;
			YS = ys;
			ZS = zs;
			XF = xf;
			YF = yf;
			ZF = zf;
			Owner = owner;
			R = DefaultR;
			G = DefaultG;
			B = DefaultB;
		}
		/// <summary>
		/// Constructs a colored line.
		/// </summary>
		/// <param name="xf">X component of first point.</param>
		/// <param name="yf">Y component of first point.</param>
		/// <param name="zf">Z component of first point.</param>
		/// <param name="xs">X component of second point.</param>
		/// <param name="ys">Y component of second point.</param>
		/// <param name="zs">Z component of second point.</param>
		/// <param name="owner">Owner index.</param>
		/// <param name="r">Red component.</param>
		/// <param name="g">Green component.</param>
		/// <param name="b">Blue component.</param>
		public Line(double xf, double yf, double zf, double xs, double ys, double zs, int owner, int r, int g, int b)
		{
			XS = xs;
			YS = ys;
			ZS = zs;
			XF = xf;
			YF = yf;
			ZF = zf;
			Owner = owner;
			R = r;
			G = g;
			B = b;			
		}
		/// <summary>
		/// Empty constructor.
		/// </summary>
		public Line() {}
	}

	/// <summary>
	/// A 3D scene.
	/// </summary>
	/// <remarks>
	/// <para>3D Scenes can be saved to files in the X3L format, and rebuilt from them.</para>
	/// <para>The X3L file format is obtained through standard .NET serialization to a text file.</para>
	/// <para>A typical X3L file has a structure like the following example:
	/// <example>
	/// <code>
	/// &lt;Scene&gt;
	///  &lt;BackColor /&gt;
	///  &lt;CameraDirectionX&gt;-1&lt;/CameraDirectionX&gt;
	///  &lt;CameraDirectionY&gt;0&lt;/CameraDirectionY&gt;
	///  &lt;CameraDirectionZ&gt;0&lt;/CameraDirectionZ&gt;
	///  &lt;CameraNormalX&gt;0&lt;/CameraNormalX&gt;
	///  &lt;CameraNormalY&gt;1&lt;/CameraNormalY&gt;
	///  &lt;CameraNormalZ&gt;0&lt;/CameraNormalZ&gt;
	///  &lt;CameraSpottingX&gt;0&lt;/CameraSpottingX&gt;
	///  &lt;CameraSpottingY&gt;0&lt;/CameraSpottingY&gt;
	///  &lt;CameraSpottingZ&gt;0&lt;/CameraSpottingZ&gt;
	///  &lt;CameraDistance&gt;10000&lt;/CameraDistance&gt;
	///  &lt;Zoom&gt;10&lt;/Zoom&gt;
	///  &lt;Points&gt;	
	///   &lt;Point&gt;
	///    &lt;X&gt;1&lt;/X&gt;
	///    &lt;Y&gt;3&lt;/Y&gt;
	///    &lt;Z&gt;-1&lt;/Z&gt;
	///    &lt;R&gt;255&lt;/R&gt;
	///    &lt;G&gt;0&lt;/G&gt;
	///    &lt;B&gt;255&lt;/B&gt;
	///    &lt;Owner&gt;0&lt;/Owner&gt;
	///   &lt;/Point&gt;	
	///  &lt;/Points&gt;
	///  &lt;Lines&gt;	
	///   &lt;Line&gt;
	///    &lt;XF&gt;5&lt;/XF&gt;
	///    &lt;YF&gt;6&lt;/YF&gt;
	///    &lt;ZF&gt;-8&lt;/ZF&gt;
	///    &lt;XS&gt;1&lt;/XS&gt;
	///    &lt;YS&gt;3&lt;/YS&gt;
	///    &lt;ZS&gt;-1&lt;/ZS&gt;
	///    &lt;R&gt;0&lt;/R&gt;
	///    &lt;G&gt;255&lt;/G&gt;
	///    &lt;B&gt;127&lt;/B&gt;
	///    &lt;Owner&gt;1&lt;/Owner&gt;
	///   &lt;/Line&gt;	
	///  &lt;/Lines&gt;
	///  &lt;OwnerSignatures&lt;
	///   &lt;string&gt;The only point.&lt;/string&gt;
	///   &lt;string&gt;The only line.&lt;/string&gt;
	///  &lt;/OwnerSignatures&lt;
	/// &lt;/Scene&gt;
	/// </code>
	/// </example>
	/// </para>
	/// </remarks>
	[Serializable]
	public class Scene
	{
		/// <summary>
		/// The background color of the scene.
		/// </summary>
		public Color BackColor;
		/// <summary>
		/// X component of camera direction.
		/// </summary>
		public double CameraDirectionX;
		/// <summary>
		/// Y component of camera direction.
		/// </summary>
		public double CameraDirectionY;
		/// <summary>
		/// Z component of camera direction.
		/// </summary>
		public double CameraDirectionZ;
		/// <summary>
		/// X component of camera normal.
		/// </summary>
		public double CameraNormalX;
		/// <summary>
		/// Y component of camera normal.
		/// </summary>
		public double CameraNormalY;
		/// <summary>
		/// Z component of camera normal.
		/// </summary>
		public double CameraNormalZ;
		/// <summary>
		/// X component of camera spotting point.
		/// </summary>
		public double CameraSpottingX;
		/// <summary>
		/// Y component of camera spotting point.
		/// </summary>
		public double CameraSpottingY;
		/// <summary>
		/// Z component of camera spotting point.
		/// </summary>
		public double CameraSpottingZ;
		/// <summary>
		/// Distance of camera from spotting point.
		/// </summary>
		public double CameraDistance;
		/// <summary>
		/// Zoom factor.
		/// </summary>
		public double Zoom;
		/// <summary>
		/// List of points.
		/// </summary>
		public Point [] Points;
		/// <summary>
		/// List of lines.
		/// </summary>
		public Line [] Lines;
		/// <summary>
		/// List of owner signatures;
		/// </summary>
		public string [] OwnerSignatures;
	}

    /// <summary>
    /// Formats for animation.
    /// </summary>
    /// <remarks>Currently only Animated GIF is supported.</remarks>
    public enum MovieFormatValue
    { 
        /// <summary>
        /// The move is an Animated GIF.
        /// </summary>
        AnimatedGif = 1     
    }

    /// <summary>
    /// An animation to be saved to file.
    /// </summary>
    /// <remarks>In the current implementation, the animation is first stored in memory and then saved to a file.</remarks>
    public class Movie
    {
        /// <summary>
        /// The format of the movie.
        /// </summary>
        public MovieFormatValue MovieFormat { get { return MovieFormatValue.AnimatedGif; } }

        /// <summary>
        /// The memory stream where the file is prepared.
        /// </summary>
        protected System.IO.MemoryStream MovieStream;

        /// <summary>
        /// Property backer for <c>TimeDelay</c>.
        /// </summary>
        protected int m_TimeDelay = 1;

        /// <summary>
        /// Time delay between frames.
        /// </summary>
        public int TimeDelay { get { return m_TimeDelay; } }

        /// <summary>
        /// Property backer for <c>Frames</c>.
        /// </summary>
        protected int m_Frames = 0;

        /// <summary>
        /// The number of frames in the movie.
        /// </summary>
        public int Frames { get { return m_Frames; } }

        /// <summary>
        /// Property backer for <c>Width</c>.
        /// </summary>
        protected int m_Width;

        /// <summary>
        /// The width of each image.
        /// </summary>
        /// <remarks>If not specified in the constructor, this field is initialized on first usage.</remarks>
        public int Width { get { return m_Width; } }

        /// <summary>
        /// Property backer for <c>Height</c>.
        /// </summary>
        protected int m_Height;

        /// <summary>
        /// The height of each image.
        /// </summary>
        /// <remarks>If not specified in the constructor, this field is initialized on first usage.</remarks>
        public int Height { get { return m_Height; } }

        /// <summary>
        /// The current size of the movie.
        /// </summary>
        public long Size { get { return (MovieStream == null) ? 0 : (MovieStream.Length + 1); } }

        /// <summary>
        /// Property backer for <c>Comment</c>.
        /// </summary>
        protected string m_Comment;

        /// <summary>
        /// Comment to be added to the movie.
        /// </summary>
        public string Comment
        {
            get { return m_Comment; }
            set { m_Comment = value; }
        }

        /// <summary>
        /// Creates a new movie.
        /// </summary>
        /// <param name="timedelay">the time delay between frames.</param>
        public Movie(int timedelay)
        {
            MovieStream = null;
            m_TimeDelay = timedelay;
            m_Width = m_Height = 0;
        }

        /// <summary>
        /// Creates a new movie.
        /// </summary>
        /// <param name="timedelay">the time delay between frames.</param>
        /// <param name="width">the acceptable image width.</param>
        /// <param name="height">the acceptable image height.</param>
        public Movie(int timedelay, int width, int height)
        {
            MovieStream = null;
            m_TimeDelay = timedelay;
            m_Width = width;
            m_Height = height;
        }

        /// <summary>
        /// Adds an image to the movie.
        /// </summary>
        /// <param name="im">the image to be added.</param>
        /// <returns>the updated number of frames.</returns>
        public int AddFrame(Image im)
        {
            if (m_Width == 0 && m_Height == 0)
            {
                m_Width = im.Width;
                m_Height = im.Height;
            }
            else if (m_Width != im.Width || m_Height != im.Height) throw new Exception("Image frame must have the same width and height as the other frames in the movie.");
            System.IO.MemoryStream tms = new System.IO.MemoryStream();
            im.Save(tms, System.Drawing.Imaging.ImageFormat.Gif);
            byte[] bys = tms.ToArray();
            if (MovieStream == null)
            {
                MovieStream = new System.IO.MemoryStream();
                MovieStream.Write(bys, 0, 781);
                MovieStream.WriteByte(33);
                MovieStream.WriteByte(255);
                MovieStream.WriteByte(11);
                MovieStream.WriteByte((byte)'N');
                MovieStream.WriteByte((byte)'E');
                MovieStream.WriteByte((byte)'T');
                MovieStream.WriteByte((byte)'S');
                MovieStream.WriteByte((byte)'C');
                MovieStream.WriteByte((byte)'A');
                MovieStream.WriteByte((byte)'P');
                MovieStream.WriteByte((byte)'E');
                MovieStream.WriteByte((byte)'2');
                MovieStream.WriteByte((byte)'.');
                MovieStream.WriteByte((byte)'0');
                MovieStream.WriteByte(3);
                MovieStream.WriteByte(1);
                MovieStream.WriteByte(0);
                MovieStream.WriteByte(0);
                MovieStream.WriteByte(0);
            }
            MovieStream.WriteByte(33);
            MovieStream.WriteByte(249);
            MovieStream.WriteByte(4);
            MovieStream.WriteByte(9); //Flags: reserved, disposal method, user input, transparent color
            MovieStream.WriteByte((byte)(m_TimeDelay & 0xff));  //Delay time low byte
            MovieStream.WriteByte((byte)((m_TimeDelay & 0xff00) >> 8));  //Delay time high byte
            MovieStream.WriteByte(255);  //Transparent color index
            MovieStream.WriteByte(0);  //Block terminator         
            if (bys[781] == 33) MovieStream.Write(bys, 789, bys.Length - 790);
            else MovieStream.Write(bys, 781, bys.Length - 782);
            return ++m_Frames;
        }

        /// <summary>
        /// Saves the movie to a file.
        /// </summary>
        /// <param name="filename">the file where the movie must be stored.</param>
        public void Save(string filename)
        {
            PrepareForWriting();
            try
            {                
                System.IO.File.WriteAllBytes(filename, MovieStream.ToArray());
            }
            finally
            {
                MovieStream.SetLength(MovieStream.Length - 1);
            }
        }

        /// <summary>
        /// Saves the movie to a stream.
        /// </summary>
        /// <param name="s">the stream where the movie has to be stored.</param>
        public void Save(System.IO.Stream s)
        {
            PrepareForWriting();
            try
            {
                MovieStream.WriteTo(s);
            }
            finally
            {
                MovieStream.SetLength(MovieStream.Length - 1);
            }            
        }

        /// <summary>
        /// Prepares the movie for writing.
        /// </summary>
        protected void PrepareForWriting()
        {
            if (m_Comment != null && m_Comment.Length > 0)
            {
                MovieStream.WriteByte(33);
                MovieStream.WriteByte(254);
                string c = ((object)m_Comment.Clone()).ToString();
                while (c.Length > 0)
                {
                    int rc = Math.Min(255, c.Length);
                    MovieStream.WriteByte((byte)rc);
                    int i;
                    for (i = 0; i < rc; i++)
                        MovieStream.WriteByte((byte)c[i]);
                    c = c.Substring(rc);
                }
                MovieStream.WriteByte(0);
            }
            MovieStream.WriteByte((byte)';');
        }

        /// <summary>
        /// Returns an animated image from the movie.
        /// </summary>
        public Image Image
        {
            get
            {
                PrepareForWriting();
                try
                {
                    System.IO.MemoryStream dupms = new System.IO.MemoryStream();
                    MovieStream.WriteTo(dupms);                    
                    return Image.FromStream(dupms);
                }
                finally
                {
                    MovieStream.SetLength(MovieStream.Length - 1);
                }
            }
        }
    }
}