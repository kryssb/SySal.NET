using System;
using System.Runtime.Serialization;
using System.Security;
[assembly:AllowPartiallyTrustedCallers]

namespace SySal
{
	namespace Executables {}

	namespace Processing {}

	namespace Services {}

	namespace BasicTypes
	{
		/// <summary>
		/// 3D vector structure
		/// </summary>
		[Serializable]
		public struct Vector
		{
			public double X;
			public double Y;
			public double Z;

            /// <summary>
            /// Inner product of two vectors.
            /// </summary>
            /// <param name="a">first vector.</param>
            /// <param name="b">second vector.</param>
            /// <returns>the inner product of the two vectors.</returns>
            public static double operator *(Vector a, Vector b)
            {
                return a.X * b.X + a.Y * b.Y + a.Z * b.Z;
            }

            /// <summary>
            /// Outer product of two vectors.
            /// </summary>
            /// <param name="a">first vector.</param>
            /// <param name="b">second vector.</param>
            /// <returns>the outer product of the two vectors.</returns>
            public static Vector operator ^(Vector a, Vector b)
            {
                SySal.BasicTypes.Vector c = new Vector();
                c.X = a.Y * b.Z - a.Z * b.Y;
                c.Y = a.Z * b.X - a.X * b.Z;
                c.Z = a.X * b.Y - a.Y * b.X;
                return c;
            }

            /// <summary>
            /// Square of the norm of the vector.
            /// </summary>
            public double Norm2
            {
                get
                {
                    return this * this;
                }
            }

            /// <summary>
            /// Scalar product.
            /// </summary>
            /// <param name="a">the scalar to multiply the vector.</param>
            /// <param name="b">the vector to be multiplied.</param>
            /// <returns>the product vector.</returns>
            public static Vector operator *(double a, Vector b)
            {
                b.X *= a;
                b.Y *= a;
                b.Z *= a;
                return b;
            }

            /// <summary>
            /// The vector with the same direction and unit norm.
            /// </summary>
            public Vector UnitVector
            {
                get
                {
                    double n = (1.0 / Math.Sqrt(Norm2));
                    return n * this;                    
                }
            }

            /// <summary>
            /// Sum of two vectors.
            /// </summary>
            /// <param name="a">first vector.</param>
            /// <param name="b">second vector.</param>
            /// <returns>the sum of the two vectors.</returns>
            public static Vector operator +(Vector a, Vector b)
            {
                a.X += b.X;
                a.Y += b.Y;
                a.Z += b.Z;
                return a;
            }

            /// <summary>
            /// Difference of two vectors.
            /// </summary>
            /// <param name="a">first vector.</param>
            /// <param name="b">second vector.</param>
            /// <returns>the difference of the two vectors.</returns>
            public static Vector operator -(Vector a, Vector b)
            {
                a.X -= b.X;
                a.Y -= b.Y;
                a.Z -= b.Z;
                return a;
            }
		}

		/// <summary>
		/// 3D line
		/// </summary>
		[Serializable]
		public struct Line
		{
			public Vector Begin;
			public Vector End;
		}

		/// <summary>
		/// 2D vector structure
		/// </summary>
		[Serializable]
		public struct Vector2
		{
			public double X;
			public double Y;

			public Vector2(Vector v)
			{
				X = v.X;
				Y = v.Y;
			}
		}

		/// <summary>
		/// 2D rectangle
		/// </summary>
		[Serializable]
		public struct Rectangle
		{
			public double MinX;
			public double MaxX;
			public double MinY;
			public double MaxY;
		}

		/// <summary>
		/// 3d cuboid
		/// </summary>
		[Serializable]
		public struct Cuboid
		{
			public double MinX;
			public double MaxX;
			public double MinY;
			public double MaxY;
			public double MinZ;
			public double MaxZ;
		}

		/// <summary>
		/// RGB color
		/// </summary>
		[Serializable]
		public struct RGBColor
		{
			public float Red;
			public float Green;
			public float Blue;
		}

		/// <summary>
		/// Colored Vector
		/// </summary>
		[Serializable]
		public struct CVector
		{
			public RGBColor C;
			public Vector V;
		}

		/// <summary>
		/// Colored Line
		/// </summary>
		[Serializable]
		public struct CLine
		{
			public RGBColor C;
			public Line L;
		}

		/// <summary>
		/// Colored Cuboid
		/// </summary>
		[Serializable]
		public struct CCuboid
		{
			public RGBColor C;
			public Cuboid Q;
		}

		/// <summary>
		/// Generic named parameter
		/// </summary>
		[Serializable]
		public struct NamedParameter
		{
			public string Name;
			public object Value;

			public NamedParameter(string n, object o) { Name = n; Value = o; }
		}

		/// <summary>
		/// Generic Identifier
		/// </summary>
		[Serializable]
		public struct Identifier
		{
			public int Part0;
			public int Part1;
			public int Part2;
			public int Part3;

			public int this[int part]
			{
				get
				{
					switch (part)
					{
						case 0:		return Part0;
						case 1:		return Part1;
						case 2:		return Part2;
						case 3:		return Part3;
					}
					throw new System.IndexOutOfRangeException("Identifier index cannot be " + part.ToString());
				}
				set
				{
					switch (part)
					{
						case 0:		Part0 = value; break;
						case 1:		Part1 = value; break;
						case 2:		Part2 = value; break;
						case 3:		Part3 = value; break;
					}
					throw new System.IndexOutOfRangeException("Identifier index cannot be " + part.ToString());
				}
			}
		}
	}

	namespace Management
	{
		/// <summary>
		/// Object configuration.
		///	</summary>
		[Serializable]
		public abstract class Configuration : ICloneable
		{
			public string Name;

			public Configuration(string confname) 
			{
				Name = confname;
			}

			public abstract object Clone();
		}

        /// <summary>
        /// Provides unified management of machine settings.
        /// </summary>
        /// <remarks>A machine setting is a set of configuration parameters that are typical of a machine rather than
        /// a user or a configuration. This is the case for the control chain of microscope stage, for a vision processor, etc.
        /// The related settings are specific to the machine and not of the user or configuration.</remarks>
        public static class MachineSettings
        {
            /// <summary>
            /// The directory that contains all configuration files.
            /// </summary>
            public static string BaseDirectory
            {
                get
                {
                    string s = System.Environment.GetEnvironmentVariable("ALLUSERSPROFILE");
                    if (!(s.EndsWith("/") || s.EndsWith("\\"))) s += "/";
                    s += "SySal";
                    return s;
                }
            }

            private static void EnsureBaseDirectoryExists()
            {
                string s = BaseDirectory;
                if (System.IO.Directory.Exists(BaseDirectory) == false)
                    System.IO.Directory.CreateDirectory(s);
            }

            /// <summary>
            /// Get/sets the configuration for a specified class.
            /// </summary>
            /// <param name="t">the type of the class for which machine settings are to be sought.</param>
            /// <returns>the configuration, if present, or a null reference if does not exists.</returns>
            public static Configuration GetSettings(Type t)
            {
                    EnsureBaseDirectoryExists();
                    try
                    {
                        string d = System.IO.File.ReadAllText(BaseDirectory + "/" + t.FullName);
                        System.Xml.Serialization.XmlSerializer xmls = new System.Xml.Serialization.XmlSerializer(t);
                        Configuration c = (Configuration)xmls.Deserialize(new System.IO.StringReader(d));
                        return c;
                    }
                    catch (Exception) 
                    {
                        return null;
                    }
            }

            public static void SetSettings(Type t, Configuration c)
            {
                    EnsureBaseDirectoryExists();
                    System.Xml.Serialization.XmlSerializer xmls = new System.Xml.Serialization.XmlSerializer(t);
                    System.IO.StringWriter w = new System.IO.StringWriter();
                    xmls.Serialize(w, c);
                    w.Flush();
                    System.IO.File.WriteAllText(BaseDirectory + "/" + t.FullName, w.ToString());
            }
        }

        /// <summary>
        /// Unified interface for classes to manage their machine settings.
        /// </summary>       
        public interface IMachineSettingsEditor
        {
            /// <summary>
            /// Opens a dialog session with the user to manage machine settings for a class.
            /// </summary>
            /// <param name="t">the type storing the machine settings.</param>
            /// <returns><c>true</c>if the settings have been changed, <c>false</c> otherwise.</returns>
            bool EditMachineSettings(Type t);
        }

		/// <summary>
		/// Connection exceptions.
		/// </summary>
		public class ConnectionException : System.Exception
		{
			private static string XMessage(System.Type [] reqtypes, bool nullallowed)
			{
				string ret = "The object must ";
				if (nullallowed) ret += "be a null reference or ";
				ret += "implement the following interface";
				if (reqtypes.Length > 0) ret += "s";
				ret += "s: ";
				foreach (Type t in reqtypes)
				{
					ret += "\r\n" + t.FullName;
				}
				ret += "\r\n" + typeof(IManageable).FullName;
				return ret;
			}

			public ConnectionException(string message) : base(message) {}

			public ConnectionException(System.Type [] reqtypes, bool nullallowed) : base(XMessage(reqtypes, nullallowed)) {}
		}


		/// <summary>
		/// Represents a connection between a client object and a server object.
		/// </summary>
		public interface IConnection
		{
			/// <summary>
			/// Connection name.
			/// </summary>
			string Name
			{
				get;
			}

			/// <summary>
			/// Object connected.
			/// </summary>
			IManageable Server
			{
				get;
				set;
			}

			/// <summary>
			/// Tells whether the connection is OK.
			/// </summary>
			bool IsOK
			{
				get;
			}
		}


		/// <summary>
		/// Connection lists are custom properties of each object.
		/// </summary>
		public interface IConnectionList
		{
			/// <summary>
			/// Accesses a connection through its index.
			/// </summary>
			IConnection this [int index]
			{
				get;
				set;
			}

			/// <summary>
			/// Accesses a connection through its name (case sensitive).
			/// </summary>
			IConnection this [string name]
			{
				get;
				set;
			}

			/// <summary>
			/// Number of connections.
			/// </summary>
			int Length
			{
				get;
			}
		}


		/// <summary>
		/// Contains basic operations for object management.
		/// </summary>
		public interface IManageable
		{	
			/// <summary>
			/// Name of the object.
			/// </summary>
			string Name
			{
				get;
				set;
			}


			/// <summary>
			/// Current object configuration.
			/// </summary>
			Configuration Config
			{
				get;
				set;
			}


			/// <summary>
			/// Uses the object to initialize / edit a configuration.
			/// </summary>
			/// <param name="c"></param>
			/// <returns></returns>
			bool EditConfiguration(ref Configuration c);

			/// <summary>
			/// The connections of this object.
			/// </summary>
			IConnectionList Connections
			{
				get;
			}


			/// <summary>
			/// Enables / disables the object monitor.
			/// </summary>
			bool MonitorEnabled
			{
				get;
				set;
			}
		}

		/// <summary>
		/// Manageable objects that can be represented graphically
		/// </summary>
		public interface IGraphicallyManageable : IManageable, IDisposable
		{
			/// <summary>
			/// X position in a GUI layout.
			/// </summary>
			int XLocation
			{
				get;
				set;
			}

			/// <summary>
			/// Y position in a GUI layout.
			/// </summary>
			int YLocation
			{
				get;
				set;
			}

			/// <summary>
			/// The object's icon.
			/// </summary>
			System.Drawing.Icon Icon
			{
				get;
			}
		}

		/// <summary>
		/// Each setup has a list of manageable objects.
		/// </summary>
		public interface IManageableList
		{
			/// <summary>
			/// Accesses a manageable object through its index.
			/// </summary>
			IManageable this [int index]
			{
				get;
				set;
			}

			/// <summary>
			/// Accesses a manageable object through its name (case sensitive).
			/// </summary>
			IManageable this [string name]
			{
				get;
				set;
			}

			/// <summary>
			/// Number of manageable objects.
			/// </summary>
			int Length
			{
				get;
			}
		}


		/// <summary>
		/// Allows objects to take control of the execution.
		/// </summary>
		public interface IExecute
		{
			/// <summary>
			/// Activates the object.
			/// </summary>
			void Execute();
		}


		/// <summary>
		/// Stores the list of server objects, along with their connections, and the indication of an "executor".
		/// </summary>
		public interface IManageableGraph
		{
			/// <summary>
			/// Name of the graph.
			/// </summary>
			string Name
			{
				get;
				set;
			}


			/// <summary>
			/// List of the objects of the graph.
			/// </summary>
			IManageableList Objects
			{
				get;
			}


			/// <summary>
			/// The object that can take control of the program execution.
			/// </summary>
			IExecute Executor
			{
				get;
				set;
			}
		}

		/// <summary>
		/// Allows objects to expose additional info about internal operations for debugging and performance monitoring.
		/// </summary>
		public interface IExposeInfo
		{
			/// <summary>
			/// Exposes / hides generation of additional info.
			/// </summary>
			bool Expose
			{
				get;
				set;
			}

			/// <summary>
			/// Gets the additional information.
			/// </summary>
			System.Collections.ArrayList ExposedInfo
			{
				get;
			}
		}	


		/// <summary>
		/// A connection that can only accept some types.
		/// </summary>
		[Serializable]
		public class FixedTypeConnection : IConnection
		{
			/// <summary>
			/// Describes the requirements for this connection.
			/// </summary>
			public struct ConnectionDescriptor
			{
				public string Name;
				public Type [] RequiredSlotTypeNames;
				public bool NullAllowed;
	
				public ConnectionDescriptor(string name, Type [] reqslottypenames, bool nullallowed)
				{
					Name = name;
					RequiredSlotTypeNames = reqslottypenames;
					NullAllowed = nullallowed;
				}
			}

			protected Type [] Interfaces;
			protected bool NullAllowed;
			protected string NameValue;
			protected object ServerObj;

			/// <summary>
			/// Builds a new FixedTypeConnection from a ConnectionDescriptor.
			/// </summary>
			/// <param name="desc"></param>
			public FixedTypeConnection(ConnectionDescriptor desc)
			{
				if (desc.RequiredSlotTypeNames.Length < 1) throw new ConnectionException("Type list must contain at least one item");
				NameValue = desc.Name;
				NullAllowed = desc.NullAllowed;
				Interfaces = (Type [])desc.RequiredSlotTypeNames.Clone();
				ServerObj = null;
			}


			/// <summary>
			/// Name of the connection.
			/// </summary>
			public string Name
			{
				get
				{
					return NameValue;
				}
			}


			/// <summary>
			/// Server object for this connection.
			/// </summary>
			public IManageable Server
			{
				get	
				{	
					return (IManageable)ServerObj;	
				}

				set
				{
					if (value != null)
						foreach (Type t in Interfaces)
							if (value.GetType().GetInterface(t.FullName, false) == null)
								throw new ConnectionException(Interfaces, NullAllowed);
					ServerObj = value;
				}
			}


			/// <summary>
			/// Tells whether the connection is OK.
			/// </summary>
			public bool IsOK
			{
				get { return NullAllowed || (ServerObj != null); }
			}

		}


		/// <summary>
		/// Connection list with fixed type constraints.
		/// </summary>
		[Serializable]
		public class FixedConnectionList : IConnectionList
		{
			/// <summary>
			/// Keeps track of the connections;
			/// </summary>
			protected FixedTypeConnection [] Connections;

			/// <summary>
			/// Builds a list of connections from a list of connection descriptors.
			/// </summary>
			/// <param name="descriptors"></param>
			public FixedConnectionList(FixedTypeConnection.ConnectionDescriptor [] descriptors)
			{
				Connections = new FixedTypeConnection[descriptors.Length];
				int i;
				for (i = 0; i < descriptors.Length; i++)
					Connections[i] = new FixedTypeConnection(descriptors[i]);
			}

			/// <summary>
			/// Accesses a connection through its index.
			/// </summary>
			public IConnection this [int index]
			{
				get	{ return Connections[index]; }
				set { throw new ConnectionException("Connections of a " + this.GetType().FullName + " cannot be changed after creation"); }
			}

			/// <summary>
			/// Accesses a connection through its name (case sensitive).
			/// </summary>
			public IConnection this [string name]
			{
				get
				{
					foreach (FixedTypeConnection c in Connections)
						if (c.Name == name) return c;
					throw new ConnectionException("Unknown connection name \"" + name + "\"");
				}

				set { throw new ConnectionException("Connections of a " + this.GetType().FullName + " cannot be changed after creation"); }
			}

			/// <summary>
			/// Number of connections.
			/// </summary>
			public int Length
			{
				get { return Connections.Length; }
			}
		}
	}
}
