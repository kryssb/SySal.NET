using System;
using System.Data;
using System.Data.Common;
using Oracle.ManagedDataAccess.Client;
using Oracle.ManagedDataAccess.Types;
using System.Collections;
using SySal.DAQSystem.Scanning;

namespace SySal.OperaDb
{
	/// <summary>
	/// Opera DB data types.
	/// </summary>
	public enum OperaDbType
	{		
		/// <summary>
		/// Integer data type.
		/// </summary>
		Int, 
		/// <summary>
		/// Long integer data type.
		/// </summary>
		Long,
		/// <summary>
		/// Single precision data type.
		/// </summary>
		Float,
		/// <summary>
		/// Double precision data type.
		/// </summary>
		Double,
		/// <summary>
		/// String data type.
		/// </summary>
		String,
		/// <summary>
		/// Date/time data type.
		/// </summary>
		DateTime,
		/// <summary>
		/// Large text data type.
		/// </summary>
		CLOB,
        /// <summary>
        /// Large binary data type.
        /// </summary>
        BLOB
	}

	/// <summary>
	/// Schema version of the DB.
	/// </summary>
	public enum DBSchemaVersion
	{ 
		/// <summary>
		/// Unknown DB schema.
		/// </summary>
		Unknown,
		/// <summary>
		/// Basic DB schema, with TB_ZONES, TB_MIPMICROTRACKS and TB_MIPBASETRACKS. Unused Reconstruction tables.
		/// </summary>
		Basic_V1, 
		/// <summary>
		/// DB schema Version 2, with TB_ZONES, TB_VIEWS, TB_MIPMICROTRACKS and TB_MIPBASETRACKS. Working Reconstruction tables.
		/// </summary>
		HasViews_V2 
	}

	/// <summary>
	/// A connection to an Opera DB.
	/// </summary>
	/// <remarks>Opera Connection objects act as wrappers for the connections in the underlying access library. </remarks>
	/// <remarks>Using OperaDbConnection objects allows developers to write code that is independent of the specific technology used to access the OPERA DB.</remarks>
	public class OperaDbConnection
	{
		/// <summary>
		/// The underlying connection object.
		/// </summary>
		protected internal object Conn;

		/// <summary>
		/// The possible connections that can be used.
		/// </summary>
		protected OracleConnection [] m_PossibleConnections;

		/// <summary>
		/// Yields access to the connection object for use with custom access functions.
		/// </summary>
		public object DbConnection { get { return Conn; } }

		/// <summary>
		/// Builds a new connection to the Opera DB specified in the dbname, using the username and password supplied.
		/// </summary>
		/// <param name="dbname">name of the Opera DB instance to connect to.</param>
		/// <param name="username">username to connect.</param>
		/// <param name="password">password to connect.</param>
		public OperaDbConnection(string dbname, string username, string password)
		{			
			Conn = null;
			string [] dbnames = dbname.Split(',');
			m_PossibleConnections = new OracleConnection[dbnames.Length];
			int i;
			for (i = 0; i < dbnames.Length; i++)
                m_PossibleConnections[i] = new OracleConnection("User Id=\"" + username + "\";Password=\"" + password + "\";Data Source=\"" + dbnames[i].Trim() + "\";Validate Connection=true;Pooling=false");
		}

        /// <summary>
        /// Builds a new connection to the Opera DB using a provider-specific connection string.
        /// </summary>
        /// <param name="connstring">the provider-specific connection string.</param>
        public OperaDbConnection(string connstring)
        {
            Conn = null;
            m_PossibleConnections = new OracleConnection[1];
            int i;
            for (i = 0; i < m_PossibleConnections.Length; i++)
                m_PossibleConnections[i] = new OracleConnection(connstring);
        }

		/// <summary>
		/// Opens the connection.
		/// </summary>
		public void Open()
		{
			//Close();
			string xx = "";
			foreach (OracleConnection c in m_PossibleConnections)
			{
				try
				{
					c.Open();
					Conn = c;
                    CheckBufferTables();
                    break;
				}
				catch (Exception x)
				{
					if (xx.Length != 0) xx += "\r\n";
					xx += x.Message;
				}
			}
			if (Conn == null) throw new Exception(xx);
		}

		/// <summary>
		/// Closes the connection.
		/// </summary>
		public void Close()
		{
			foreach (OracleConnection c in m_PossibleConnections)
			{
				if (c != null) c.Close();
			}
			Conn = null;
		}

		/// <summary>
		/// Opens a transaction.
		/// </summary>
		/// <returns></returns>
		public OperaDbTransaction BeginTransaction()
		{
			return new OperaDbTransaction(((OracleConnection)Conn).BeginTransaction(System.Data.IsolationLevel.ReadCommitted), this);
		}

		/// <summary>
		/// The string format used for dates.
		/// </summary>
		public static readonly string TimeFormat = "'DD/MM/YYYY HH24:MI:SS'";

		/// <summary>
		/// Formats a Date/Time according to the required TimeFormat.
		/// </summary>
		/// <param name="d">the Date/Time to be formatted.</param>
		/// <returns>the formatted date/time string.</returns>
		public static string ToTimeFormat(System.DateTime d)
		{
			return d.Day.ToString() + "/" + d.Month.ToString() + "/" + d.Year.ToString() + " " + d.Hour.ToString() + ":" + d.Minute.ToString() + ":" + d.Second.ToString();
		}

		internal static Oracle.ManagedDataAccess.Client.OracleDbType ToOracleType(SySal.OperaDb.OperaDbType opdbtype)
		{
			switch (opdbtype)
			{
				case SySal.OperaDb.OperaDbType.Int: return Oracle.ManagedDataAccess.Client.OracleDbType.Int32;
				case SySal.OperaDb.OperaDbType.Long: return Oracle.ManagedDataAccess.Client.OracleDbType.Int64;
				case SySal.OperaDb.OperaDbType.Float: return Oracle.ManagedDataAccess.Client.OracleDbType.Double;
				case SySal.OperaDb.OperaDbType.Double: return Oracle.ManagedDataAccess.Client.OracleDbType.Double;
				case SySal.OperaDb.OperaDbType.String: return Oracle.ManagedDataAccess.Client.OracleDbType.Varchar2;
				case SySal.OperaDb.OperaDbType.DateTime: return Oracle.ManagedDataAccess.Client.OracleDbType.TimeStamp;
				case SySal.OperaDb.OperaDbType.CLOB: return Oracle.ManagedDataAccess.Client.OracleDbType.Clob;
                case SySal.OperaDb.OperaDbType.BLOB: return Oracle.ManagedDataAccess.Client.OracleDbType.Blob;
			}
			throw new Exception("The type cannot be mapped to an Oracle type!");
		}

		/// <summary>
		/// Member data on which the DBSchemaVersion property relies. 
		/// Set to "Unknown" at the beginning: the DB Schema for zones is checked upon the first access.
		/// </summary>
		protected DBSchemaVersion m_DBSchemaVersion = DBSchemaVersion.Unknown;

		/// <summary>
		/// The current DB schema version.
		/// </summary>
		public DBSchemaVersion DBSchemaVersion 
		{ 
			get 
			{ 
				if (m_DBSchemaVersion == DBSchemaVersion.Unknown) CheckDBSchemaVersion();
				return m_DBSchemaVersion; 
			} 
		}

		void CheckDBSchemaVersion()
		{
			if (SySal.OperaDb.Convert.ToInt32(new OperaDbCommand("SELECT COUNT(*) FROM ALL_TABLES WHERE OWNER = 'OPERA' AND TABLE_NAME = 'TB_VIEWS'", this, null).ExecuteScalar()) == 1)
			{
				m_DBSchemaVersion = DBSchemaVersion.HasViews_V2;
			}
			else
			{
				m_DBSchemaVersion = DBSchemaVersion.Basic_V1;
			}
		}

        void CheckBufferTables()
        {
            int c = SySal.OperaDb.Convert.ToInt32(new OperaDbCommand("select count(*) from all_procedures where owner = 'OPERA' and object_name = 'LP_DATABUFFER_FLUSH'", this).ExecuteScalar());
            m_HasBufferTables = (c > 0);
        }

        /// <summary>
        /// Member data on which HasBufferTables relies. Can be accessed by derived classes.
        /// </summary>
        protected bool m_HasBufferTables;

        /// <summary>
        /// <c>True</c> if buffer tables are present in this DB, false otherwise.
        /// </summary>
        public bool HasBufferTables
        {
            get
            {
                return m_HasBufferTables;
            }
        }

        /// <summary>
        /// Flushes buffer tables. Is automatically called by <see cref="SySal.OperaDb.OperaDbTransaction.Commit"/>.
        /// </summary>
        public void FlushBufferTables()
        {
            if (HasBufferTables) new Oracle.ManagedDataAccess.Client.OracleCommand("CALL OPERA.LP_DATABUFFER_FLUSH()", (Oracle.ManagedDataAccess.Client.OracleConnection)Conn).ExecuteNonQuery();
        }
	}

	/// <summary>
	/// General conversion class for DB return types.
	/// </summary>
    public class Convert
    {
        /// <summary>
        /// Converts the object to Int64. If the object is null an exception is thrown.
        /// </summary>
        /// <param name="o">object to be converted.</param>
        /// <returns>the value of the object.</returns>
        public static long ToInt64(object o)
        {
            if (o == null) throw new Exception("Null value cannot be converted.");
            if (o.GetType() == typeof(OracleDecimal)) return ((OracleDecimal)o).ToInt64();
            return System.Convert.ToInt64(o);
        }

        /// <summary>
        /// Converts the object to UInt64. If the object is null an exception is thrown.
        /// </summary>
        /// <param name="o">object to be converted.</param>
        /// <returns>the value of the object.</returns>
        public static ulong ToUInt64(object o)
        {
            if (o == null) throw new Exception("Null value cannot be converted.");
            if (o.GetType() == typeof(OracleDecimal)) return Convert.ToUInt64(((OracleDecimal)o).ToInt64());
            return System.Convert.ToUInt64(o);
        }

        /// <summary>
        /// Converts the object to Int32. If the object is null an exception is thrown.
        /// </summary>
        /// <param name="o">object to be converted.</param>
        /// <returns>the value of the object.</returns>
        public static int ToInt32(object o)
        {
            if (o == null) throw new Exception("Null value cannot be converted.");
            if (o.GetType() == typeof(OracleDecimal)) return ((OracleDecimal)o).ToInt32();
            return System.Convert.ToInt32(o);
        }

        /// <summary>
        /// Converts the object to UInt32. If the object is null an exception is thrown.
        /// </summary>
        /// <param name="o">object to be converted.</param>
        /// <returns>the value of the object.</returns>
        public static uint ToUInt32(object o)
        {
            if (o == null) throw new Exception("Null value cannot be converted.");
            if (o.GetType() == typeof(OracleDecimal)) return Convert.ToUInt32(((OracleDecimal)o).ToInt32()); ;
            return System.Convert.ToUInt32(o);
        }

        /// <summary>
        /// Converts the object to Int16. If the object is null an exception is thrown.
        /// </summary>
        /// <param name="o">object to be converted.</param>
        /// <returns>the value of the object.</returns>
        public static short ToInt16(object o)
        {
            if (o == null) throw new Exception("Null value cannot be converted.");
            if (o.GetType() == typeof(OracleDecimal)) return ((OracleDecimal)o).ToInt16();
            return System.Convert.ToInt16(o);
        }

        /// <summary>
        /// Converts the object to UInt16. If the object is null an exception is thrown.
        /// </summary>
        /// <param name="o">object to be converted.</param>
        /// <returns>the value of the object.</returns>
        public static ushort ToUInt16(object o)
        {
            if (o == null) throw new Exception("Null value cannot be converted.");
            if (o.GetType() == typeof(OracleDecimal)) return Convert.ToUInt16(((OracleDecimal)o).ToInt16()); ;
            return System.Convert.ToUInt16(o);
        }

        /// <summary>
        /// Converts the object to Boolean. If the object is null an exception is thrown.
        /// </summary>
        /// <param name="o">object to be converted.</param>
        /// <returns>the value of the object.</returns>
        public static bool ToBoolean(object o)
        {
            if (o == null) throw new Exception("Null value cannot be converted.");
            return System.Convert.ToBoolean(o);
        }

        /// <summary>
        /// Converts the object to Double. If the object is null an exception is thrown.
        /// </summary>
        /// <param name="o">object to be converted.</param>
        /// <returns>the value of the object.</returns>
        public static double ToDouble(object o)
        {
            if (o == null) throw new Exception("Null value cannot be converted.");
            if (o.GetType() == typeof(System.Decimal)) return System.Convert.ToDouble(o);
            return System.Convert.ToDouble(o);
        }

        /// <summary>
        /// Converts a BLOB object to an array of bytes. If the object is null an exception is thrown.
        /// </summary>
        /// <param name="o">BLOB object to be converted.</param>
        /// <returns>the byte content of the BLOB.</returns>
        public static byte[] ToBytes(object o)
        {
            Oracle.ManagedDataAccess.Types.OracleBlob blob = (Oracle.ManagedDataAccess.Types.OracleBlob)o;
            byte[] bytes = new byte[blob.Length];
            blob.Read(bytes, 0, (int)blob.Length);
            return bytes;
        }
    }

	/// <summary>
	/// A transaction in an Opera DB context.
	/// </summary>
	/// <remarks>Opera Transaction objects act as wrappers for the transactions in the underlying access library. </remarks>
	/// <remarks>Using OperaDbTransaction objects allows developers to write code that is independent of the specific technology used to access the OPERA DB.</remarks>
	public class OperaDbTransaction
	{
		/// <summary>
		/// The underlying transaction object.
		/// </summary>
		internal object Trans;

        /// <summary>
        /// Keeps track of the owner DB connection.
        /// </summary>
        protected OperaDbConnection m_Conn;

        internal OperaDbTransaction(object trans, OperaDbConnection conn) 
		{
			Trans = trans;
            m_Conn = conn;
		}

		/// <summary>
		/// Commits the transaction.
		/// </summary>
		public void Commit()
		{
            if (m_Conn.HasBufferTables) new OperaDbCommand("CALL OPERA.LP_DATABUFFER_FLUSH()", m_Conn).ExecuteNonQuery();
			((OracleTransaction)Trans).Commit();
		}

		/// <summary>
		/// Cancels the transaction.
		/// </summary>
		public void Rollback()
		{
			((OracleTransaction)Trans).Rollback();
		}	
	}

    /// <summary>
    /// A DataReader for OperaDb.
    /// </summary>
    public class OperaDbDataReader : IDisposable
    {
        /// <summary>
        /// The underlying DataReader object.
        /// </summary>
        protected internal OracleDataReader m_DataReader;

        /// <summary>
        /// Yields access to the underlying DataReader object.
        /// </summary>
        public object DataReader { get { return m_DataReader; } }

        /// <summary>
        /// Creates a new OperaDbDataReader using a DataReader that has been previously created.
        /// </summary>
        /// <param name="r">the DataReader object to be wrapped.</param>
        protected internal OperaDbDataReader(object r)
        {
            m_DataReader = (OracleDataReader)r;                        
        }

        /// <summary>
        /// Closes the DataReader.
        /// </summary>
        public void Close()
        {
            m_DataReader.Close();
        }

        /// <summary>
        /// Reads a new row from the result set. It must be called to get the first row.
        /// </summary>
        /// <returns>true if the row has been read, false if no more rows are available.</returns>
        public bool Read()
        {
            return m_DataReader.Read();
        }

        /// <summary>
        /// Reads a new row from the result set. It must be called to get the first row. This method must be used with REF CURSORs.
        /// </summary>
        /// <returns>true if the row has been read, false if no more rows are available.</returns>
        public bool NextResult()
        {
            return m_DataReader.NextResult();
        }

        /// <summary>
        /// Checks whether a field is <c>DBNull</c>.
        /// </summary>
        /// <param name="i">the number of the field to be checked.</param>
        /// <returns><c>true</c> if the field is <c>DBNull</c>, <c>false</c> otherwise.</returns>
        public bool IsDBNull(int i)
        {
            return m_DataReader.IsDBNull(i);
        }

        /// <summary>
        /// <c>true</c> if the DataReader is closed, <c>false</c> otherwise.
        /// </summary>        
        public bool IsClosed
        {
            get
            {
                return m_DataReader.IsClosed;
            }
        }

        /// <summary>
        /// <c>true</c> if the statement returned any row, <c>false</c> otherwise.
        /// </summary>
        public bool HasRows
        {
            get { return m_DataReader.HasRows; }
        }

        /// <summary>
        /// The number of fields in the result set.
        /// </summary>
        public int FieldCount
        {
            get { return m_DataReader.FieldCount; }
        }

        /// <summary>
        /// The number of visible fields in the result set.
        /// </summary>
        public int VisibleFieldCount
        {
            get { return m_DataReader.FieldCount/*.VisibleFieldCount*/; }
        }

        /// <summary>
        /// The number of records updated, inserted or deleted.
        /// </summary>
        public int RecordsAffected
        {
            get { return m_DataReader.RecordsAffected; }
        }

        /// <summary>
        /// Gets the name of the specified column.
        /// </summary>
        /// <param name="i">the zero-based index of the column for which the name is sought.</param>
        /// <returns>the name of the column.</returns>
        public string GetName(int i)
        {
            return m_DataReader.GetName(i);
        }

        /// <summary>
        /// Gets the ordinal number of a column, given the name.
        /// </summary>
        /// <param name="name">the name of the column.</param>
        /// <returns>the zero-based index of the column with the specified name.</returns>
        public int GetOrdinal(string name)
        {
            return m_DataReader.GetOrdinal(name);
        }

        /// <summary>
        /// Gets a string from the specified column.
        /// </summary>
        /// <param name="i">the zero-based index of the column for which the string value is sought.</param>
        /// <returns>the value of the column as a string.</returns>
        public string GetString(int i)
        {
            return m_DataReader.GetString(i);
        }

        /// <summary>
        /// Gets an <c>Int32</c> from the specified column.
        /// </summary>
        /// <param name="i">the zero-based index of the column for which the <c>Int32</c> value is sought.</param>
        /// <returns>the value of the column as an <c>Int32</c>.</returns>
        public int GetInt32(int i)
        {
            return SySal.OperaDb.Convert.ToInt32(m_DataReader.GetDecimal(i));
        }

        /// <summary>
        /// Gets an <c>Int64</c> from the specified column.
        /// </summary>
        /// <param name="i">the zero-based index of the column for which the <c>Int64</c> value is sought.</param>
        /// <returns>the value of the column as an <c>Int64</c>.</returns>
        public long GetInt64(int i)
        {
            return SySal.OperaDb.Convert.ToInt64(m_DataReader.GetDecimal(i));
        }

        /// <summary>
        /// Gets a <c>Double</c> from the specified column.
        /// </summary>
        /// <param name="i">the zero-based index of the column for which the <c>Double</c> value is sought.</param>
        /// <returns>the value of the column as a <c>Double</c>.</returns>
        public double GetDouble(int i)
        {
            return SySal.OperaDb.Convert.ToDouble(m_DataReader.GetDecimal(i));
        }

        /// <summary>
        /// Gets a <c>DateTime</c> from the specified column.
        /// </summary>
        /// <param name="i">the zero-based index of the column for which the <c>DateTime</c> value is sought.</param>
        /// <returns>the value of the column as a <c>DateTime</c>.</returns>
        public DateTime GetDateTime(int i)
        {
            return m_DataReader.GetDateTime(i);
        }

        /// <summary>
        /// Gets an <c>object</c> from the specified column in its native format.
        /// </summary>
        /// <param name="i">the zero-based index of the column for which the value is sought.</param>
        /// <returns>the value of the column as an <c>object</c>.</returns>
        public object GetValue(int i)
        {
            return m_DataReader.GetValue(i);
        }

        #region IDisposable Members

        /// <summary>
        /// Disposes the object.
        /// </summary>
        public void Dispose()
        {
            m_DataReader.Dispose();
        }

        #endregion
    }

	/// <summary>
	/// A DataAdapter for OperaDb.
	/// </summary>
	public class OperaDbDataAdapter
	{
		/// <summary>
		/// The underlying DataAdapter object.
		/// </summary>
		protected internal OracleDataAdapter m_DataAdapter;

		/// <summary>
		/// Yields access to the underlying DataAdapter object.
		/// </summary>
		public object DataAdapter { get { return m_DataAdapter; } }

		/// <summary>
		/// Builds a new OperaDbDataAdapter.
		/// </summary>
		/// <param name="commandtext">SQL query string.</param>
		/// <param name="conn">DB to connect to.</param>
		public OperaDbDataAdapter(string commandtext, OperaDbConnection conn)
		{
			m_DataAdapter = new OracleDataAdapter(commandtext, (OracleConnection)conn.Conn);
		}

		/// <summary>
		/// Builds a new OperaDbDataAdapter.
		/// </summary>
		/// <param name="commandtext">SQL query string.</param>
		/// <param name="conn">DB to connect to.</param>
		/// <param name="trans">transaction context in which the query is to be executed.</param>
		public OperaDbDataAdapter(string commandtext, OperaDbConnection conn, OperaDbTransaction trans)
		{
			m_DataAdapter = new OracleDataAdapter(commandtext, (OracleConnection)conn.Conn);
		}

		/// <summary>
		/// Fills a DataSet.
		/// </summary>
		/// <param name="ds">data set to be filled.</param>
		/// <returns>the number of rows affected.</returns>
		public int Fill(DataSet ds)
		{
			return m_DataAdapter.Fill(ds);
		}
	}

	/// <summary>
	/// A parameter for an SQL command.
	/// </summary>
	public class OperaDbParameter
	{
		/// <summary>
		/// The underlying parameter object.
		/// </summary>
		protected internal OracleParameter m_Parameter;

		/// <summary>
		/// Builds an OperaDbParameter wrapper for an Oracle parameter.
		/// </summary>
		/// <param name="parameter">the parameter object being wrapped.</param>
		protected internal OperaDbParameter(OracleParameter parameter)
		{
			m_Parameter = parameter;
		}
			
		/// <summary>
		/// Builds a new OperaDbParameter.
		/// </summary>
		/// <param name="name">parameter name.</param>
		/// <param name="dbtype">the data type of the parameter.</param>
		/// <param name="dir">the parameter direction</param>
		public OperaDbParameter(string name, SySal.OperaDb.OperaDbType dbtype, System.Data.ParameterDirection dir)
		{
			m_Parameter = new OracleParameter(name, OperaDbConnection.ToOracleType(dbtype), dir);
			if (dbtype == SySal.OperaDb.OperaDbType.String) m_Parameter.Size = 4096;
		}

		/// <summary>
		/// Builds a new OperaDbParameter.
		/// </summary>
		/// <param name="name">parameter name.</param>
		/// <param name="dbtype">the data type of the parameter.</param>
		/// <param name="size">the data size of the parameter.</param>
		/// <param name="dir">the parameter direction</param>
		public OperaDbParameter(string name, SySal.OperaDb.OperaDbType dbtype, int size, System.Data.ParameterDirection dir)
		{
			m_Parameter = new OracleParameter(name, OperaDbConnection.ToOracleType(dbtype), size, dir);
			m_Parameter.Size = size;
		}

		/// <summary>
		/// The value of the parameter.
		/// </summary>
		public object Value
		{
			get
			{
				return m_Parameter.Value;
			}
			set
			{
				m_Parameter.Value = value;
			}
		}
	}

	/// <summary>
	/// A collection of parameters for an SQL statement.
	/// </summary>
	public class OperaDbParameterCollection
	{
		/// <summary>
		/// The underlying ParameterCollection object.
		/// </summary>
		protected internal OracleParameterCollection m_ParameterCollection;

		/// <summary>
		/// Builds an OperaDbParameterCollection wrapper for an Oracle parameter collection.
		/// </summary>
		/// <param name="parametercollection">the parameter collection object being wrapped.</param>
		protected internal OperaDbParameterCollection(OracleParameterCollection parametercollection)
		{
			m_ParameterCollection = parametercollection;
		}

		/// <summary>
		/// Adds a new parameter.
		/// </summary>
		/// <param name="name">parameter name.</param>
		/// <param name="dbtype">parameter type.</param>
		/// <param name="size">parameter size.</param>
		/// <param name="dir">parameter direction.</param>
		/// <returns>the parameter that has been created and added to the collection.</returns>
		public OperaDbParameter Add(string name, OperaDbType dbtype, int size, System.Data.ParameterDirection dir)
		{
			Oracle.ManagedDataAccess.Client.OracleParameter p = m_ParameterCollection.Add(name, OperaDbConnection.ToOracleType(dbtype), size, dir);
			p.Size = size;
			return new OperaDbParameter(p); 
		}

		/// <summary>
		/// Adds a new parameter.
		/// </summary>
		/// <param name="name">parameter name.</param>
		/// <param name="dbtype">parameter type.</param>		
		/// <param name="dir">parameter direction.</param>
		/// <returns>the parameter that has been created and added to the collection.</returns>
		public OperaDbParameter Add(string name, OperaDbType dbtype, System.Data.ParameterDirection dir)
		{
			Oracle.ManagedDataAccess.Client.OracleParameter p = m_ParameterCollection.Add(name, OperaDbConnection.ToOracleType(dbtype), dir);
			if (dbtype == SySal.OperaDb.OperaDbType.String) p.Size = 256;
			return new OperaDbParameter(p); 
		}

		/// <summary>
		/// Accesses the i-th parameter in the collection.
		/// </summary>
		public OperaDbParameter this[int i]
		{
			get
			{
				return new OperaDbParameter(m_ParameterCollection[i]);
			}
		}

		/// <summary>
		/// Accesses a parameter in the collection through its name.
		/// </summary>
		public OperaDbParameter this[string n]
		{
			get
			{
				return new OperaDbParameter(m_ParameterCollection[n]);
			}
		}
	}

	/// <summary>
	/// A command to be executed in an Opera DB.
	/// </summary>
	public class OperaDbCommand
	{
		/// <summary>
		/// The underlying command object.
		/// </summary>
		protected internal OracleCommand m_Command;

		/// <summary>
		/// Yields access to the underlying Command object.
		/// </summary>
		public object Command { get { return m_Command; } }

		/// <summary>
		/// The underlying Connection object.
		/// </summary>
		protected internal OperaDbConnection m_Connection;

		/// <summary>
		/// Gets / sets the command connection.
		/// </summary>
		public OperaDbConnection Connection 
		{
			get { return m_Connection; }
			set { m_Connection = value; m_Command.Connection = (m_Connection == null) ? null : (OracleConnection)m_Connection.Conn; }
		}

		/// <summary>
		/// The underlying Transaction object.
		/// </summary>
		protected internal OperaDbTransaction m_Transaction;

		/// <summary>
		/// Gets / sets the command transaction context.
		/// </summary>
		public OperaDbTransaction Transaction 
		{
			get { return m_Transaction; }
			set { m_Transaction = value; }
		}

		/// <summary>
		/// Builds a new command object.
		/// </summary>
		/// <param name="commandtext">SQL query text for the command.</param>
		public OperaDbCommand(string commandtext)
		{
			m_Command = new OracleCommand(commandtext);
		}

		/// <summary>
		/// Builds a new command object.
		/// </summary>
		/// <param name="commandtext">SQL query text for the command.</param>
		/// <param name="conn">connection to be used for the query.</param>
		public OperaDbCommand(string commandtext, OperaDbConnection conn)
		{
			m_Command = new OracleCommand(commandtext, (OracleConnection)((m_Connection = conn).Conn));	
		}

		/// <summary>
		/// Builds a new command object.
		/// </summary>
		/// <param name="commandtext">SQL query text for the command.</param>
		/// <param name="conn">connection to be used for the query.</param>
		/// <param name="trans">transaction context to be used for the query.</param>
		public OperaDbCommand(string commandtext, OperaDbConnection conn, OperaDbTransaction trans)
		{
			m_Command = new OracleCommand(commandtext, (OracleConnection)((m_Connection = conn).Conn));	
		}

		/// <summary>
		/// Executes a query without return values.
		/// </summary>
		/// <returns>the number of rows affected.</returns>
		public int ExecuteNonQuery()
		{
			return m_Command.ExecuteNonQuery();
		}

		/// <summary>
		/// Executes a query that returns a single value.
		/// </summary>
		/// <returns>the return object.</returns>
		public object ExecuteScalar()
		{
			return m_Command.ExecuteScalar();
		}

        /// <summary>
        /// Executes the command and returns a reader to get the result set.
        /// </summary>
        /// <returns>the <c>OperaDbDataReader</c> objects that reads the result set.</returns>
        public OperaDbDataReader ExecuteReader()
        {
            return new OperaDbDataReader(m_Command.ExecuteReader());
        }

		/// <summary>
		/// Prepares (compiles) a command for faster execution.
		/// </summary>
		public void Prepare()
		{
			m_Command.Prepare();
		}

		/// <summary>
		/// The parameters for the SQL statement.
		/// </summary>
		public OperaDbParameterCollection Parameters
		{
			get
			{
				return new OperaDbParameterCollection(m_Command.Parameters);
			}
		}

		/// <summary>
		/// Number of items for array binding.
		/// </summary>
		public int ArrayBindCount
		{
			get
			{
				return m_Command.ArrayBindCount;
			}
			set
			{
				m_Command.ArrayBindCount = value;
			}
		}
	}


	namespace TotalScan
	{
		/// <summary>
		/// Index for a Segment related to an m.i.p. microtrack in the DB. Stores information about zone, side, and microtrack Id.
		/// </summary>
		public class DBMIPMicroTrackIndex : SySal.TotalScan.Index
		{
			/// <summary>
			/// The signature of the DBMIPMicroTrackIndex class.
			/// </summary>
			public static readonly int Signature = 3;

			/// <summary>
			/// Registers the Index factory for DBMIPMicroTrackIndex.
			/// </summary>
			public static void RegisterFactory()
			{
				SySal.TotalScan.Index.RegisterFactory(new IndexFactory(Signature, 14, new SySal.TotalScan.Index.dCreateFromReader(CreateFromReader), new SySal.TotalScan.Index.dSaveToWriter(SaveToWriter)));				
			}

			/// <summary>
			/// Member data on which the ZoneId property relies.
			/// </summary>
			protected long m_ZoneId;
			/// <summary>
			/// The Id of the zone the microtrack belongs to.
			/// </summary>
			public long ZoneId { get { return m_ZoneId; } }
			/// <summary>
			/// Member data on which the Side property relies.
			/// </summary>
			protected short m_Side;
			/// <summary>
			/// The side of the MIP Microtrack.
			/// </summary>
			public short Side { get { return m_Side; } }
			/// <summary>
			/// Member data on which the Id property relies.
			/// </summary>
			protected int m_Id;
			/// <summary>
			/// The Id of the microtrack in its side.
			/// </summary>
			public int Id { get { return m_Id; } }

			/// <summary>
			/// Constructs a DBMIPMicroTrackIndex.
			/// </summary>
			/// <param name="zoneid">the id of the zone of the microtrack.</param>
			/// <param name="side">the side of the microtrack.</param>
			/// <param name="id">the index of the microtrack.</param>
			public DBMIPMicroTrackIndex(long zoneid, short side, int id)
			{
				m_ZoneId = zoneid;
				m_Side = side;
				m_Id = id;
			}

			/// <summary>
			/// Saves a DBMIPMicroTrackIndex to a BinaryWriter.
			/// </summary>
			/// <param name="b">the BinaryWriter to be used for saving.</param>
			public override void Write(System.IO.BinaryWriter b)
			{
				b.Write(m_ZoneId);
				b.Write(m_Side);
				b.Write(m_Id);
			}

			/// <summary>
			/// Converts a DBMIPMicroTrackIndex to text form.
			/// </summary>
			/// <returns>a string of the form "zoneid\side\id".</returns>
			public override string ToString()
			{
				return m_ZoneId.ToString() + @"\" + m_Side.ToString() + @"\" + m_Id.ToString();
			}

			/// <summary>
			/// Saves a DBMIPMicroTrackIndex to a BinaryWriter.
			/// </summary>
			/// <param name="i">the index to be saved.</param>
			/// <param name="w">the BinaryWriter to be used for writing.</param>
			public static void SaveToWriter(SySal.TotalScan.Index i, System.IO.BinaryWriter w)
			{
				((DBMIPMicroTrackIndex)i).Write(w);
			}

			/// <summary>
			/// Reads a DBMIPMicroTrackIndex from a BinaryReader.
			/// </summary>
			/// <param name="r">the BinaryReader to read the DBMIPMicroTrackIndex from.</param>
			/// <returns>the index read from the BinaryReader.</returns>
			public static SySal.TotalScan.Index CreateFromReader(System.IO.BinaryReader r)
			{
				return new DBMIPMicroTrackIndex(r.ReadInt64(), r.ReadInt16(), r.ReadInt32());
			}

			/// <summary>
			/// Returns the IndexFactory for the MIPMicroTrackIndex class.
			/// </summary>
			public override IndexFactory Factory { get { return new IndexFactory(Signature, 14, new SySal.TotalScan.Index.dCreateFromReader(CreateFromReader), new SySal.TotalScan.Index.dSaveToWriter(SaveToWriter)); } }

			#region ICloneable Members

			/// <summary>
			/// Clones a DBMIPMicroTrackIndex.
			/// </summary>
			/// <returns>the cloned object.</returns>
			public override object Clone()
			{
				return new DBMIPMicroTrackIndex(m_ZoneId, m_Side, m_Id);
			}

			#endregion			

			public override bool Equals(object obj)
			{
				if (obj.GetType() != this.GetType()) return false;
				DBMIPMicroTrackIndex x = (DBMIPMicroTrackIndex)obj;
				return x.ZoneId == this.m_ZoneId && x.Side == this.m_Side && x.Id == this.m_Id;
			}

			public override int GetHashCode()
			{
				return ((int)m_ZoneId) + m_Side + m_Id;
			}

		}

		/// <summary>
		/// A segment stored in an Opera DB.
		/// </summary>
		public class Segment : SySal.TotalScan.Segment
		{
			private static bool m_RegisteredIndexFactories = RegisterIndexFactories();

			private static bool RegisterIndexFactories()
			{
				DBMIPMicroTrackIndex.RegisterFactory();
				return true;
			}
			
			/// <summary>
			/// Member data on which the DB_Id property relies. Can be accessed by derived classes.
			/// </summary>
			protected internal long m_DB_Id;
			/// <summary>
			/// The Id of the segment in the Opera DB.
			/// </summary>
			public long DB_Id { get { return m_DB_Id; } }

			/// <summary>
			/// Protected constructor. Prevents users from creating OperaDb.Segments without deriving the class. Is implicitly called by constructors in derived classes.
			/// </summary>
			protected internal Segment() {}

			internal Segment(SySal.Tracking.MIPEmulsionTrackInfo info, SySal.TotalScan.Index ix) : base(info, ix) {}

			internal void SetTrackOwner(Track t, int posintrack) { m_TrackOwner = t; m_PosInTrack = posintrack; }
		}

		/// <summary>
		/// A layer stored in an Opera DB.
		/// </summary>
		public class Layer : SySal.TotalScan.Layer
		{
			/// <summary>
			/// Protected constructor. Prevents users from creating OperaDb.Layers without deriving the class. Is implicitly called by constructors in derived classes.
			/// </summary>
			protected internal Layer() {}
			internal Layer(int id, long brickid, int sheetid, short side, SySal.BasicTypes.Vector Ref_Center) : base(id, brickid, sheetid, side, Ref_Center)
			{
				
			}
			internal Layer(int id, long brickid, int sheetid, short side, SySal.BasicTypes.Vector Ref_Center, double DownstreamZ, double UpstreamZ) : base(id, brickid, sheetid, side, Ref_Center, DownstreamZ, UpstreamZ)
			{
				
			}
			internal void SetAlignmentData(SySal.TotalScan.AlignmentData a)
			{
				m_AlignmentData = a;
			}
		}

		/// <summary>
		/// Index class for attributes identified with a name and a process operation id.
		/// </summary>
		public class DBNamedAttributeIndex : SySal.TotalScan.Index
		{
			/// <summary>
			/// The signature of the DBNamedAttributeIndex class.
			/// </summary>
			public static readonly int Signature = 52;

			const int NameLen = 32;

			/// <summary>
			/// Registers the Index factory for DBNamedAttributeIndex.
			/// </summary>
			public static void RegisterFactory()
			{
				SySal.TotalScan.Index.RegisterFactory(new IndexFactory(Signature, NameLen * 2 + 4, new SySal.TotalScan.Index.dCreateFromReader(CreateFromReader), new SySal.TotalScan.Index.dSaveToWriter(SaveToWriter)));
			}

			/// <summary>
			/// Member data on which the Name property relies.
			/// </summary>
			protected string m_Name;

			/// <summary>
			/// Name of the attribute.
			/// </summary>
			public string Name { get { return (string)(m_Name.Clone()); } }

			/// <summary>
			/// Member data on which the ProcOpId property relies.
			/// </summary>
			protected long m_ProcOpId;

			/// <summary>
			/// Process operation Id of the attribute.
			/// </summary>
			public long ProcOpId { get { return m_ProcOpId; } }

			/// <summary>
			/// Constructs a DBNamedAttributeIndex from an attribute name (max 32 chars) and a process operation id.
			/// </summary>
			/// <param name="name">the name to be assigned to the attribute</param>
			/// <param name="procopid">the id of the process operation that computed the attribute.</param>
			public DBNamedAttributeIndex(string name, long procopid)
			{
				m_Name = name.PadRight(NameLen, ' ').Substring(0, NameLen).Trim();
				if (m_Name.Length == 0) throw new Exception("Name must be a non-null string. Spaces are trimmed, but are preserved between words.");
				m_ProcOpId = procopid;
			}

			/// <summary>
			/// Saves a DBNamedAttributeIndex to a BinaryWriter.
			/// </summary>
			/// <param name="b">the BinaryWriter to be used for saving.</param>
			public override void Write(System.IO.BinaryWriter b)
			{
				char [] chars = m_Name.PadRight(NameLen, ' ').ToCharArray(0, NameLen);
				b.Write(chars);
				b.Write(m_ProcOpId);
			}

			/// <summary>
			/// Converts the DBNamedAttributeIndex to a text form.
			/// </summary>
			/// <returns>the Name and the ProcOpId in text form.</returns>
			public override string ToString()
			{
				return m_Name + " " + m_ProcOpId;
			}

			/// <summary>
			/// Saves a DBNamedAttributeIndex to a BinaryWriter.
			/// </summary>
			/// <param name="i">the index to be saved. Must be a DBNamedAttributeIndex.</param>
			/// <param name="w">the BinaryWriter to be used for saving.</param>
			public static void SaveToWriter(SySal.TotalScan.Index i, System.IO.BinaryWriter w)
			{
				((DBNamedAttributeIndex)i).Write(w);
			}

			/// <summary>
			/// Reads a DBNamedAttributeIndex from a BinaryReader.
			/// </summary>
			/// <param name="r">the BinaryReader to read from.</param>
			/// <returns>the DBNamedAttributeIndex read from the stream.</returns>
			public static SySal.TotalScan.Index CreateFromReader(System.IO.BinaryReader r)
			{
				return new DBNamedAttributeIndex(new string(r.ReadChars(NameLen)).Trim(), r.ReadInt64());
			}

			/// <summary>
			/// Returns the IndexFactory for a DBNamedAttributeIndex.
			/// </summary>
			public override IndexFactory Factory { get { return new IndexFactory(Signature, NameLen * 2 + 4, new SySal.TotalScan.Index.dCreateFromReader(CreateFromReader), new SySal.TotalScan.Index.dSaveToWriter(SaveToWriter)); } }

			#region ICloneable Members

			/// <summary>
			/// Clones this DBNamedAttributeIndex.
			/// </summary>
			/// <returns>a clone of the DBNamedAttributeIndex.</returns>
			public override object Clone()
			{
				return new DBNamedAttributeIndex(m_Name, m_ProcOpId);
			}

			#endregion

			public override bool Equals(object obj)
			{
				if (obj.GetType() != this.GetType()) return false;
				DBNamedAttributeIndex x = (DBNamedAttributeIndex)obj;
				return (String.Compare(x.Name, this.m_Name, true) == 0) && x.ProcOpId == m_ProcOpId;
			}

			public override int GetHashCode()
			{
				return m_Name.GetHashCode() + (int)m_ProcOpId;
			}

		}

		/// <summary>
		/// A track stored in an Opera DB.
		/// </summary>
		public class Track : SySal.TotalScan.Track
		{
			/// <summary>
			/// Protected constructor. Prevents users from creating OperaDb.Tracks without deriving the class. Is implicitly called by constructors in derived classes.
			/// </summary>
			protected internal Track() {}
			internal Track(int id, Segment [] segs) : base(id)
			{
				m_Upstream_PosX_Updated = m_Upstream_PosY_Updated = m_Upstream_SlopeX_Updated = m_Upstream_SlopeY_Updated = 
					m_Downstream_PosX_Updated = m_Downstream_PosY_Updated = m_Downstream_SlopeX_Updated = m_Downstream_SlopeY_Updated = false;
				Segments = segs;
				int i;
				for (i = 0; i < segs.Length; i++)					
					segs[i].SetTrackOwner(this, i);
			}

			internal void SetUpVertex(Vertex v) { this.m_Upstream_Vertex = v; }
			internal void SetDownVertex(Vertex v) { this.m_Downstream_Vertex = v; }
		}

		/// <summary>
		/// A vertex stored in an Opera DB.
		/// </summary>
		public class Vertex : SySal.TotalScan.Vertex
		{
			/// <summary>
			/// Protected constructor. Prevents users from creating OperaDb.Vertices without deriving the class. Is implicitly called by constructors in derived classes.
			/// </summary>
			protected internal Vertex() {}
			internal Vertex(int id) : base(id)
			{
				Tracks = new Track[0];
			}

			internal void Add(SySal.TotalScan.Track t)
			{
				SySal.TotalScan.Track [] newtks = new Track[Tracks.Length + 1];
				int i;
				for (i = 0; i < Tracks.Length; i++)
					newtks[i] = Tracks[i];
				newtks[i] = t;
				Tracks = newtks;
			}
		}

		/// <summary>
		/// A volume stored in an Opera DB.
		/// </summary>
		public class Volume : SySal.TotalScan.Volume
		{
			/// <summary>
			/// Layer list in an OperaDb.Volume.
			/// </summary>
			protected class DBLayerList : SySal.TotalScan.Volume.LayerList
			{
				/// <summary>
				/// Constructor. Builds a DBLayerList with the specified Layer array.
				/// </summary>
				/// <param name="layers">array of layers to initialize the DBLayerList.</param>
				public DBLayerList(Layer [] layers) { this.Items = layers; }
			}

			/// <summary>
			/// Track list in an OperaDb.Volume.
			/// </summary>
			protected class DBTrackList : SySal.TotalScan.Volume.TrackList
			{
				/// <summary>
				/// Constructor. Builds a DBTrackList with the specified Track array.
				/// </summary>
				/// <param name="tracks">array of Tracks to initialize the DBTrackList.</param>
				public DBTrackList(Track [] tracks) { this.Items = tracks; }
			}

			/// <summary>
			/// Vertex list in an OperaDb.Volume.
			/// </summary>
			protected class DBVertexList : SySal.TotalScan.Volume.VertexList
			{
				/// <summary>
				/// Constructor. Builds a DBVertexList with the specified Track array.
				/// </summary>
				/// <param name="vertices">array of Vertices to initialize the DBVertexList.</param>
				public DBVertexList(Vertex [] vertices) { this.Items = vertices; }
			}

			/// <summary>
			/// Member data on which the DB_Id property relies. Can be accessed by derived classes.
			/// </summary>
			protected internal long m_DB_Id;
			/// <summary>
			/// The Id of the volume in the Opera DB.
			/// </summary>
			public long DB_Id { get { return m_DB_Id; } }
			/// <summary>
			/// Member data on which the Option_Id property relies. Can be accessed by derived classes.
			/// </summary>
			protected internal long m_Option_Id;
			/// <summary>
			/// The Option Id of the volume in the Opera DB.
			/// </summary>
			public long Option_Id { get { return m_Option_Id; } }

			/// <summary>
			/// Protected constructor. Prevents users from creating OperaDb.Volumes without deriving the class. Is implicitly called by constructors in derived classes.
			/// </summary>
			protected internal Volume() {}
			internal Volume(long db_id, long opt_id)
			{
				m_DB_Id = db_id;
				m_Option_Id = opt_id;
			}


			/// <summary>
			/// Reads a TotalScan volume with a specified id from an Opera DB.
			/// Optionally restores also segment references to base tracks.
			/// </summary>
			/// <param name="conn">open connection to the Opera DB.</param>
			/// <param name="trans">transaction context to be used. Can be null.</param>
			/// <param name="brick_id">DB identification number of the brick where the volume is found.</param>
			/// <param name="db_id">DB identification number of the TotalScan volume to be restored.</param>
			/// <param name="opt_id">Option Id. If zero, volume tracks and vertices are not read. If nonzero, volume tracks and vertices of the corresponding reconstruction option are read.</param>
			public Volume(OperaDbConnection conn, OperaDbTransaction trans, long brick_id, long db_id, long opt_id)
			{
				OracleConnection oracleconn = (OracleConnection)conn.Conn;
				DataSet ds = new DataSet();
				OracleDataAdapter da;
                da = new OracleDataAdapter("SELECT TB_ALIGNED_SIDES.ID_PLATE, TB_ALIGNED_SIDES.SIDE, REFX, REFY, REFZ, TB_ALIGNED_SIDES.DOWNZ, TB_ALIGNED_SIDES.UPZ, TXX, TXY, TYX, TYY, TDX, TDY, TDZ, SDX, SDY, SMX, SMY FROM TB_ALIGNED_SLICES INNER JOIN TB_ALIGNED_SIDES ON (TB_ALIGNED_SIDES.ID_EVENTBRICK = " + brick_id + " AND TB_ALIGNED_SLICES.ID_EVENTBRICK = " + brick_id + " AND TB_ALIGNED_SIDES.ID_RECONSTRUCTION = " + db_id + " AND TB_ALIGNED_SLICES.ID_RECONSTRUCTION = " + db_id + " AND TB_ALIGNED_SLICES.ID_PLATE = TB_ALIGNED_SIDES.ID_PLATE) ORDER BY DOWNZ DESC", oracleconn);
				da.Fill(ds);
				DataTable dt = ds.Tables[0];
				DataRow dr = null;

				int i, j, n, m;
                short side;
				Layer [] layers = new Layer[n = dt.Rows.Count];
				for (i = 0; i < n; i++)
				{
					dr = dt.Rows[i];
					SySal.BasicTypes.Vector u = new SySal.BasicTypes.Vector();
					u.X = Convert.ToDouble(dr[2]);
					u.Y = Convert.ToDouble(dr[3]);
                    side = Convert.ToInt16(dr[1]);
					u.Z = Convert.ToDouble(dr[(side == 1) ? 6 : 5]);
					Layer l = layers[i] = new Layer(i, brick_id, Convert.ToInt32(dr[0]), Convert.ToInt16(dr[1]), u, Convert.ToDouble(dr[5]), Convert.ToDouble(dr[6]));
					SySal.TotalScan.AlignmentData a = new SySal.TotalScan.AlignmentData();
					a.AffineMatrixXX = Convert.ToDouble(dr[7]);
					a.AffineMatrixXY = Convert.ToDouble(dr[8]);
					a.AffineMatrixYX = Convert.ToDouble(dr[9]);
					a.AffineMatrixYY = Convert.ToDouble(dr[10]);
					a.TranslationX = Convert.ToDouble(dr[11]);
					a.TranslationY = Convert.ToDouble(dr[12]);
					a.TranslationZ = Convert.ToDouble(dr[13]);
					a.SAlignDSlopeX = Convert.ToDouble(dr[14]);
					a.SAlignDSlopeY = Convert.ToDouble(dr[15]);
					a.DShrinkX = Convert.ToDouble(dr[16]);
					a.DShrinkY = Convert.ToDouble(dr[17]);
					l.SetAlignmentData(a);
				}
	
				if (opt_id == 0)
				{
					m_Tracks = new DBTrackList(new SySal.OperaDb.TotalScan.Track[0]);
					m_Vertices = new DBVertexList(new SySal.OperaDb.TotalScan.Vertex[0]);
					for (i = 0; i < n; i++)
					{
						ds = new DataSet();
						da = new OracleDataAdapter("SELECT ID_ZONE, ID, POSX, POSY, SLOPEX, SLOPEY, SIGMA, GRAINS, AREASUM FROM TB_ALIGNED_MIPMICROTRACKS WHERE (ID_EVENTBRICK = " + brick_id + " AND ID_RECONSTRUCTION = " + db_id + " AND ID_PLATE = " + layers[i].SheetId + " AND SIDE = " + layers[i].Side + ")", oracleconn);
						da.Fill(ds); dt = ds.Tables[0]; m = dt.Rows.Count;
						Segment [] segments = new Segment[m];
						for (j = 0; j < m; j++)
						{
							dr = dt.Rows[j];
							SySal.Tracking.MIPEmulsionTrackInfo info = new SySal.Tracking.MIPEmulsionTrackInfo();
							info.Intercept.X = Convert.ToDouble(dr[2]);
							info.Intercept.Y = Convert.ToDouble(dr[3]);
							info.Intercept.Z = layers[i].RefCenter.Z;
							info.Slope.X = Convert.ToDouble(dr[4]);
							info.Slope.Y = Convert.ToDouble(dr[5]);
							info.Slope.Z = 1.0;
							info.Sigma = Convert.ToDouble(dr[6]);
							info.Count = Convert.ToUInt16(dr[7]);
							info.AreaSum = Convert.ToUInt32(dr[8]);
							info.TopZ = layers[i].DownstreamZ;
							info.BottomZ = layers[i].UpstreamZ;
							segments[j] = new Segment(info, new DBMIPMicroTrackIndex(Convert.ToInt64(dr[0]), layers[i].Side, Convert.ToInt32(dr[1])));
						}
						layers[i].AddSegments(segments);
					}
					m_Layers = new DBLayerList(layers);
				}
				else
				{
					System.Collections.ArrayList [] pretrackids = new System.Collections.ArrayList[Convert.ToInt32(new OracleCommand("SELECT COUNT(*) FROM TB_VOLUMETRACKS WHERE ID_EVENTBRICK = " + brick_id + " AND ID_RECONSTRUCTION = " + db_id + " AND ID_OPTION = " + opt_id, oracleconn).ExecuteScalar())];
					for (i = 0; i < pretrackids.Length; i++)
						pretrackids[i] = new System.Collections.ArrayList();
					for (i = 0; i < n; i++)
					{
						ds = new DataSet();
                        da = new OracleDataAdapter("SELECT IDZ, IDMU, POSX, POSY, SLOPEX, SLOPEY, SIGMA, GRAINS, AREASUM, ID_VOLUMETRACK FROM " +
                            "(SELECT TB_ALIGNED_MIPMICROTRACKS.ID_ZONE as IDZ, TB_ALIGNED_MIPMICROTRACKS.ID as IDMU, POSX, POSY, SLOPEX, SLOPEY, SIGMA, GRAINS, AREASUM FROM TB_ALIGNED_MIPMICROTRACKS WHERE ID_EVENTBRICK = " + brick_id + " AND ID_RECONSTRUCTION = " + db_id + " AND ID_PLATE = " + layers[i].SheetId + " AND SIDE = " + layers[i].Side + ") " +
                            "LEFT JOIN (SELECT ID_ZONE, ID_MIPMICROTRACK, TB_VOLTKS_ALIGNMUTKS.ID_VOLUMETRACK FROM TB_VOLTKS_ALIGNMUTKS WHERE ID_EVENTBRICK = " + brick_id + " AND ID_RECONSTRUCTION = " + db_id + " AND ID_OPTION = " + opt_id + " AND ID_PLATE = " + layers[i].SheetId + " AND SIDE = " + layers[i].Side + ") ON " + 
							"(IDZ = ID_ZONE AND IDMU = ID_MIPMICROTRACK)", oracleconn);
						da.Fill(ds); dt = ds.Tables[0]; m = dt.Rows.Count;                         
						Segment [] segments = new Segment[m];
						for (j = 0; j < m; j++)
						{
							dr = dt.Rows[j];
							SySal.Tracking.MIPEmulsionTrackInfo info = new SySal.Tracking.MIPEmulsionTrackInfo();
							info.Intercept.X = Convert.ToDouble(dr[2]);
							info.Intercept.Y = Convert.ToDouble(dr[3]);
							info.Intercept.Z = layers[i].RefCenter.Z;
							info.Slope.X = Convert.ToDouble(dr[4]);
							info.Slope.Y = Convert.ToDouble(dr[5]);
							info.Slope.Z = 1.0;
							info.Sigma = Convert.ToDouble(dr[6]);
							info.Count = Convert.ToUInt16(dr[7]);
							info.AreaSum = Convert.ToUInt32(dr[8]);
							info.TopZ = layers[i].DownstreamZ;
							info.BottomZ = layers[i].UpstreamZ;
							segments[j] = new Segment(info, new DBMIPMicroTrackIndex(Convert.ToInt64(dr[0]), layers[i].Side, Convert.ToInt32(dr[1])));
							if (dr[9] != System.DBNull.Value)
								pretrackids[Convert.ToInt32(dr[9]) - 1].Add(segments[j]);
						}
						layers[i].AddSegments(segments);
					}			
					m_Layers = new DBLayerList(layers);
		
					SySal.OperaDb.TotalScan.Track [] tks = new SySal.OperaDb.TotalScan.Track[pretrackids.Length];
					for (i = 0; i < tks.Length; i++)
						tks[i] = new Track(i, (SySal.OperaDb.TotalScan.Segment [])(pretrackids[i].ToArray(typeof(SySal.OperaDb.TotalScan.Segment))));

					m_Tracks = new DBTrackList(tks);
					pretrackids = null;

					SySal.OperaDb.TotalScan.Vertex [] vtxs = new SySal.OperaDb.TotalScan.Vertex[Convert.ToInt32(new OracleCommand("SELECT COUNT(*) FROM TB_VERTICES WHERE ID_EVENTBRICK = " + brick_id + " AND ID_RECONSTRUCTION = " + db_id + " AND ID_OPTION = " + opt_id, oracleconn).ExecuteScalar())];
					for (i = 0; i < vtxs.Length; i++)
						vtxs[i] = new Vertex(i);
					
					ds = new DataSet();
					new OracleDataAdapter("SELECT ID, ID_DOWNSTREAMVERTEX, ID_UPSTREAMVERTEX FROM TB_VOLUMETRACKS " + 
						"WHERE (ID_EVENTBRICK = " + brick_id + " AND ID_RECONSTRUCTION = " + db_id + " AND " + 
						"ID_OPTION = " + opt_id + " AND (ID_DOWNSTREAMVERTEX IS NOT NULL OR ID_UPSTREAMVERTEX IS NOT NULL))", oracleconn).Fill(ds);
					dt = ds.Tables[0];
					foreach (System.Data.DataRow drt in dt.Rows)
					{
						Track t = tks[Convert.ToInt32(drt[0]) - 1];
						if (drt[1] != System.DBNull.Value)
						{
							i = Convert.ToInt32(drt[1]);
							t.SetDownstreamVertex(vtxs[i - 1]);
							vtxs[i - 1].Add(t);
						}
						if (drt[2] != System.DBNull.Value)
						{
							i = Convert.ToInt32(drt[2]);
							t.SetUpstreamVertex(vtxs[i - 1]);
							vtxs[i - 1].Add(t);
						}
					}
					m_Vertices = new DBVertexList(vtxs);

					ds = new DataSet();
					new OracleDataAdapter("SELECT ID, ID_PROCESSOPERATION, NAME, VALUE FROM TB_VERTICES_ATTR " + 
						"WHERE ID_EVENTBRICK = " + brick_id + " AND ID_RECONSTRUCTION = " + db_id + " AND " + 
						"ID_OPTION = " + opt_id, oracleconn).Fill(ds);
					dt = ds.Tables[0];
					foreach (System.Data.DataRow drt in dt.Rows)
						m_Vertices[Convert.ToInt32(drt[0]) - 1].SetAttribute(new DBNamedAttributeIndex(drt[2].ToString(), Convert.ToInt64(drt[1])), Convert.ToDouble(drt[3]));

					ds = new DataSet();
					new OracleDataAdapter("SELECT ID, ID_PROCESSOPERATION, NAME, VALUE FROM TB_VOLUMETRACKS_ATTR " + 
						"WHERE ID_EVENTBRICK = " + brick_id + " AND ID_RECONSTRUCTION = " + db_id + " AND " + 
						"ID_OPTION = " + opt_id, oracleconn).Fill(ds);
					dt = ds.Tables[0];
					foreach (System.Data.DataRow drt in dt.Rows)
						m_Tracks[Convert.ToInt32(drt[0]) - 1].SetAttribute(new DBNamedAttributeIndex(drt[2].ToString(), Convert.ToInt64(drt[1])), Convert.ToDouble(drt[3]));

				}				
			}

			const int BatchSize = 500;
			static long [] a_idbrick = new long[BatchSize];
			static long [] a_idrec = new long[BatchSize];
			static long [] a_idopt = new long[BatchSize];
			static int [] a_idpl = new int[BatchSize];
			static double [] a_minx = new double[BatchSize];
			static double [] a_maxx = new double[BatchSize];
			static double [] a_miny = new double[BatchSize];
			static double [] a_maxy = new double[BatchSize];
			static double [] a_refx = new double[BatchSize];
			static double [] a_refy = new double[BatchSize];
			static double [] a_refz = new double[BatchSize];
			static double [] a_downz = new double[BatchSize];
			static double [] a_upz = new double[BatchSize];
			static double [] a_txx = new double[BatchSize];
			static double [] a_txy = new double[BatchSize];
			static double [] a_tyx = new double[BatchSize];
			static double [] a_tyy = new double[BatchSize];
			static double [] a_tdx = new double[BatchSize];
			static double [] a_tdy = new double[BatchSize];
			static double [] a_tdz = new double[BatchSize];
			static int [] a_side = new int[BatchSize];
			static double [] a_sdx = new double[BatchSize];
			static double [] a_sdy = new double[BatchSize];
			static double [] a_smx = new double[BatchSize];
			static double [] a_smy = new double[BatchSize];
			static long [] a_idzone = new long[BatchSize];
			static int [] a_id = new int[BatchSize];
			static int [] a_idvoltk = new int[BatchSize];
			static double [] a_posx = new double[BatchSize];
			static double [] a_posy = new double[BatchSize];
			static double [] a_slopex = new double[BatchSize];
			static double [] a_slopey = new double[BatchSize];
			static double [] a_sigma = new double[BatchSize];
			static int [] a_grains = new int[BatchSize];
			static int [] a_areasum = new int[BatchSize];
			static int [] a_iddwvtx = new int[BatchSize];
			static int [] a_idupvtx = new int[BatchSize];
			static long [] a_idproc = new long[BatchSize];
			static string [] a_name = new string[BatchSize];
			static double [] a_value = new double[BatchSize];

			static OracleCommand cmdLslice = null;
			static OracleCommand cmdLside = null;
			static OracleCommand cmdS = null;
			static OracleCommand cmdT = null;
			static OracleCommand cmdTS = null;
			static OracleCommand cmdV = null;
			static OracleCommand cmdTA = null;
			static OracleCommand cmdVA = null;

			/// <summary>
			/// Saves the geometry (i.e. Layers and Segments) of a TotalScan volume to an Opera DB and returns the associated id.
			/// It is assumed that the Segment indices are objects of DBMIPMicroTrackIndex type (or a derived one), so that links can be established.
			/// NullIndex indices are not allowed, as well as other Index types.
			/// Tracks and vertices (i.e. the topology) are not saved.
			/// </summary>
			/// <param name="v">the TotalScan Volume object to be saved to the Opera DB.</param>
			/// <param name="id_brick">the DB identification number of the brick.</param>
			/// <param name="id_proc">the DB identification number of the process operation.</param>
			/// <param name="conn">DB connection to the Opera Db.</param>
			/// <param name="trans">DB transaction to be used. Should not be null, since the TotalScan reconstruction usually needs several tables.</param>
			/// <returns>the DB identification number that has been associated to the reconstruction.</returns>
			public static long SaveGeometry(SySal.TotalScan.Volume v, long id_brick, long id_proc, OperaDbConnection conn, OperaDbTransaction trans)
			{
				OracleConnection oracleconn = (OracleConnection)conn.Conn;
				int i, b, j, k, n, m, h;
				long idrec;
				
				if (v.Layers.Length % 2 == 1) throw new Exception("Only volumes with an even number of layers can be saved.");
				for (i = 0; i < v.Layers.Length; i += 2)				
					if (v.Layers[i].Side != 1 || v.Layers[i + 1].Side != 2 || v.Layers[i + 1].SheetId != v.Layers[i].SheetId) throw new Exception("Layers must come in pairs, with downstream side (=1) before the upstream side (=2).\r\nWrong sequence found (Sheet,Side)->(Sheet,Side)=(" +
                        v.Layers[i].Side + "," + v.Layers[i].SheetId + ")->(" + v.Layers[i + 1].Side + "," + v.Layers[i + 1].SheetId + ")");


				OracleCommand cmdinsv = new OracleCommand("INSERT INTO TB_RECONSTRUCTIONS (ID_EVENTBRICK, ID_PROCESSOPERATION) VALUES (" + id_brick + ", " + id_proc + ") RETURNING ID INTO :newid", oracleconn);
				cmdinsv.Parameters.Add("newid", OracleDbType.Int64, System.Data.ParameterDirection.Output);
				cmdinsv.ExecuteNonQuery();
				idrec = Convert.ToInt64(cmdinsv.Parameters[0].Value);
				
				lock (a_idbrick)
				{
					for (i = 0; i < BatchSize; i++)
					{
						a_idbrick[i] = id_brick;
						a_idrec[i] = idrec;						
					}
					if (cmdLslice == null)
					{
						cmdLslice = new OracleCommand("INSERT INTO TB_ALIGNED_SLICES (ID_EVENTBRICK, ID_RECONSTRUCTION, ID_PLATE, MINX, MAXX, MINY, MAXY, REFX, REFY, REFZ, DOWNZ, UPZ, TXX, TXY, TYX, TYY, TDX, TDY, TDZ) VALUES (:p_id_eventbrick, :p_id_reconstruction, :p_id_plate, :p_minx, :p_maxx, :p_miny, :p_maxy, :p_refx, :p_refy, :p_refz, :p_downz, :p_upz, :p_txx, :p_txy, :p_tyx, :p_tyy, :p_tdx, :p_tdy, :p_tdz)", oracleconn);
						cmdLslice.Parameters.Add("p_id_eventbrick", OracleDbType.Int64, System.Data.ParameterDirection.Input).Value = a_idbrick;
						cmdLslice.Parameters.Add("p_id_reconstruction", OracleDbType.Int64, System.Data.ParameterDirection.Input).Value = a_idrec;
						cmdLslice.Parameters.Add("p_id_plate", OracleDbType.Int32, System.Data.ParameterDirection.Input).Value = a_idpl;
						cmdLslice.Parameters.Add("p_minx", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = a_minx;
						cmdLslice.Parameters.Add("p_maxx", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = a_maxx;
						cmdLslice.Parameters.Add("p_miny", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = a_miny;
						cmdLslice.Parameters.Add("p_maxy", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = a_maxy;
						cmdLslice.Parameters.Add("p_refx", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = a_refx;
						cmdLslice.Parameters.Add("p_refy", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = a_refy;
						cmdLslice.Parameters.Add("p_refz", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = a_refz;
						cmdLslice.Parameters.Add("p_downz", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = a_downz;
						cmdLslice.Parameters.Add("p_upz", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = a_upz;
						cmdLslice.Parameters.Add("p_txx", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = a_txx;
						cmdLslice.Parameters.Add("p_txy", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = a_txy;
						cmdLslice.Parameters.Add("p_tyx", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = a_tyx;
						cmdLslice.Parameters.Add("p_tyy", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = a_tyy;
						cmdLslice.Parameters.Add("p_tdx", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = a_tdx;
						cmdLslice.Parameters.Add("p_tdy", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = a_tdy;
						cmdLslice.Parameters.Add("p_tdz", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = a_tdz;						
					}
					else cmdLslice.Connection = oracleconn;
					cmdLslice.Prepare();

					n = v.Layers.Length / 2;
					for (i = 0; i < n; i += BatchSize)
					{
						b = i + BatchSize; if (b > n) b = n;
						for (j = i; j < b; j++)
						{
							k = j - i;
							SySal.TotalScan.Layer lay = v.Layers[j * 2];
							a_idpl[k] = lay.SheetId;
							a_minx[k] = v.Extents.MinX;
							a_maxx[k] = v.Extents.MaxX;
							a_miny[k] = v.Extents.MinY;
							a_maxy[k] = v.Extents.MaxY;
							a_refx[k] = lay.RefCenter.X;
							a_refy[k] = lay.RefCenter.Y;
							a_refz[k] = lay.RefCenter.Z;
							a_downz[k] = lay.DownstreamZ;
							a_upz[k] = v.Layers[j * 2 + 1].UpstreamZ;
							SySal.TotalScan.AlignmentData al = lay.AlignData;
							a_txx[k] = al.AffineMatrixXX;
							a_txy[k] = al.AffineMatrixXY;
							a_tyx[k] = al.AffineMatrixYX;
							a_tyy[k] = al.AffineMatrixYY;
							a_tdx[k] = al.TranslationX;
							a_tdy[k] = al.TranslationY;
							a_tdz[k] = al.TranslationZ;
						}
						cmdLslice.ArrayBindCount = (b - i);
						cmdLslice.ExecuteNonQuery();
					}
					cmdLslice.Connection = null;
										

					if (cmdLside == null)
					{
						cmdLside = new OracleCommand("INSERT INTO TB_ALIGNED_SIDES (ID_EVENTBRICK, ID_RECONSTRUCTION, ID_PLATE, SIDE, DOWNZ, UPZ, SDX, SDY, SMX, SMY) VALUES (:p_id_eventbrick, :p_id_reconstruction, :p_id_plate, :p_side, :p_downz, :p_upz, :p_sdx, :p_sdy, :p_smx, :p_smy)", oracleconn);
						cmdLside.Parameters.Add("p_id_eventbrick", OracleDbType.Int64, System.Data.ParameterDirection.Input).Value = a_idbrick;
						cmdLside.Parameters.Add("p_id_reconstruction", OracleDbType.Int64, System.Data.ParameterDirection.Input).Value = a_idrec;
						cmdLside.Parameters.Add("p_id_plate", OracleDbType.Int32, System.Data.ParameterDirection.Input).Value = a_idpl;
						cmdLside.Parameters.Add("p_side", OracleDbType.Int32, System.Data.ParameterDirection.Input).Value = a_side;
						cmdLside.Parameters.Add("p_downz", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = a_downz;
						cmdLside.Parameters.Add("p_upz", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = a_upz;
						cmdLside.Parameters.Add("p_sdx", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = a_txx;
						cmdLside.Parameters.Add("p_sdy", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = a_txy;
						cmdLside.Parameters.Add("p_smx", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = a_tyx;
						cmdLside.Parameters.Add("p_smy", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = a_tyy;
					}
					else cmdLside.Connection = oracleconn;
					cmdLside.Prepare();

					n = v.Layers.Length;
					for (i = 0; i < n; i += BatchSize)
					{
						b = i + BatchSize; if (b > n) b = n;
						for (j = i; j < b; j++)
						{
							k = j - i;
							SySal.TotalScan.Layer lay = v.Layers[j];
							a_idpl[k] = lay.SheetId;
							a_side[k] = lay.Side;
							a_downz[k] = lay.DownstreamZ;
							a_upz[k] = lay.UpstreamZ;
							SySal.TotalScan.AlignmentData al = lay.AlignData;
							a_sdx[k] = al.SAlignDSlopeX;
							a_sdy[k] = al.SAlignDSlopeY;
							a_smx[k] = al.DShrinkX;
							a_smy[k] = al.DShrinkY;
						}
						cmdLside.ArrayBindCount = (b - i);
						cmdLside.ExecuteNonQuery();
					}
					cmdLside.Connection = null;


					if (cmdS == null)
					{
						cmdS = new OracleCommand("INSERT INTO TB_ALIGNED_MIPMICROTRACKS (ID_EVENTBRICK, ID_RECONSTRUCTION, ID_PLATE, ID_ZONE, SIDE, ID, POSX, POSY, SLOPEX, SLOPEY, SIGMA, GRAINS, AREASUM) VALUES (:p_id_eventbrick, :p_id_reconstruction, :p_id_plate, :p_id_zone, :p_side, :p_id, :p_posx, :p_posy, :p_slopex, :p_slopey, :p_sigma, :p_grains, :p_areasum)", oracleconn);
						cmdS.Parameters.Add("p_id_eventbrick", OracleDbType.Int64, System.Data.ParameterDirection.Input).Value = a_idbrick;
						cmdS.Parameters.Add("p_id_reconstruction", OracleDbType.Int64, System.Data.ParameterDirection.Input).Value = a_idrec;
						cmdS.Parameters.Add("p_id_plate", OracleDbType.Int32, System.Data.ParameterDirection.Input).Value = a_idpl;
						cmdS.Parameters.Add("p_id_zone", OracleDbType.Int64, System.Data.ParameterDirection.Input).Value = a_idzone;
						cmdS.Parameters.Add("p_side", OracleDbType.Int32, System.Data.ParameterDirection.Input).Value = a_side;
						cmdS.Parameters.Add("p_id", OracleDbType.Int32, System.Data.ParameterDirection.Input).Value = a_id;
						cmdS.Parameters.Add("p_posx", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = a_posx;
						cmdS.Parameters.Add("p_posy", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = a_posy;
						cmdS.Parameters.Add("p_slopex", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = a_slopex;
						cmdS.Parameters.Add("p_slopey", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = a_slopey;
						cmdS.Parameters.Add("p_sigma", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = a_sigma;
						cmdS.Parameters.Add("p_grains", OracleDbType.Int32, System.Data.ParameterDirection.Input).Value = a_grains;
						cmdS.Parameters.Add("p_areasum", OracleDbType.Int32, System.Data.ParameterDirection.Input).Value = a_areasum;
					}
					else cmdS.Connection = oracleconn;
					cmdS.Prepare();


					n = v.Layers.Length;
					double dz;
					for (h = 0; h < n; h++)
					{
						SySal.TotalScan.Layer lay = v.Layers[h];
						m = lay.Length;
						for (i = 0; i < m; i += BatchSize)
						{
							b = i + BatchSize; if (b > m) b = m;
							for (j = i; j < b; j++)
							{
								k = j - i;
								SySal.TotalScan.Segment s = lay[j];
								SySal.Tracking.MIPEmulsionTrackInfo info = s.Info;
								DBMIPMicroTrackIndex ix = (DBMIPMicroTrackIndex)(s.Index);
								a_idpl[k] = lay.SheetId;
								a_idzone[k] = ix.ZoneId;
								a_side[k] = ix.Side;
								a_id[k] = ix.Id;
                                dz = lay.RefCenter.Z - info.Intercept.Z;
								//dz = (ix.Side == 1) ? (lay.UpstreamZ - info.Intercept.Z) : (lay.DownstreamZ - info.Intercept.Z);
								a_posx[k] = info.Intercept.X + info.Slope.X * dz;
								a_posy[k] = info.Intercept.Y + info.Slope.Y * dz;
								a_slopex[k] = info.Slope.X;
								a_slopey[k] = info.Slope.Y;
								a_sigma[k] = info.Sigma;
								a_grains[k] = info.Count;
								a_areasum[k] = (int)info.AreaSum;
							}
							cmdS.ArrayBindCount = (b - i);
							cmdS.ExecuteNonQuery();
						}
					}
					cmdS.Connection = null;

				}

				return idrec;
			}

			/// <summary>
			/// Saves the topology (i.e., Tracks and Vertices) of a TotalScan volume to an Opera DB and returns the associated id.
			/// It is assumed that the geometry (i.e. Layers and Segments) have already been saved by a call from SaveGeometry, which should also provide the reconstruction Id.
			/// Also track attributes and vertex attributes are saved. It is assumed that attribute index types are only DBNamedAttributeIndex type.
			/// </summary>
			/// <param name="v">the TotalScan Volume object to be saved to the Opera DB.</param>
			/// <param name="id_brick">the DB identification number of the brick.</param>
			/// <param name="id_rec">the DB identification number of the reconstruction for which topology information is to be saved.</param>
			/// <param name="id_opt">the option identifier.</param>
			/// <param name="conn">DB connection to the Opera Db.</param>
			/// <param name="trans">DB transaction to be used. Should not be null, since the TotalScan reconstruction usually needs several tables.</param>
            /// <returns>the DB identification number of the newly created option.</returns>
			public static long SaveTopology(SySal.TotalScan.Volume v, long id_brick, long id_rec, long id_opt, OperaDbConnection conn, OperaDbTransaction trans)
			{
				OracleConnection oracleconn = (OracleConnection)conn.Conn;
				int i, j, b, n, m, h, k, o;
				SySal.TotalScan.Attribute [] attrs = null;
				lock (a_idbrick)
				{
                    OracleCommand cmdinsv = new OracleCommand("INSERT INTO TB_RECONSTRUCTION_OPTIONS (ID_EVENTBRICK, ID_RECONSTRUCTION, GEOMETRICAL_PROBABILITY, PHYSICAL_PROBABILITY) VALUES (" + id_brick + ", " + id_rec + ",0,0) RETURNING ID INTO :newid", oracleconn);
                    cmdinsv.Parameters.Add("newid", OracleDbType.Int64, System.Data.ParameterDirection.Output);
                    cmdinsv.ExecuteNonQuery();
                    id_opt = Convert.ToInt64(cmdinsv.Parameters[0].Value);

					for (i = 0; i < BatchSize; i++)
					{
						a_idbrick[i] = id_brick;
						a_idrec[i] = id_rec;						
						a_idopt[i] = id_opt;	
					}

					if (cmdV == null)
					{
						cmdV = new OracleCommand("INSERT INTO TB_VERTICES (ID_EVENTBRICK, ID_RECONSTRUCTION, ID_OPTION, ID) VALUES (:p_id_eventbrick, :p_id_reconstruction, :p_id_opt, :p_id)", oracleconn);
						cmdV.Parameters.Add("p_id_eventbrick", OracleDbType.Int64, System.Data.ParameterDirection.Input).Value = a_idbrick;
						cmdV.Parameters.Add("p_id_reconstruction", OracleDbType.Int64, System.Data.ParameterDirection.Input).Value = a_idrec;
						cmdV.Parameters.Add("p_id_opt", OracleDbType.Int64, System.Data.ParameterDirection.Input).Value = a_idopt;
						cmdV.Parameters.Add("p_id", OracleDbType.Int32, System.Data.ParameterDirection.Input).Value = a_id;
					}
					else cmdV.Connection = oracleconn;
					cmdV.Prepare();

					n = v.Vertices.Length;
					for (i = 0; i < n; i += BatchSize)
					{
						b = i + BatchSize; if (b > n) b = n;
						for (j = i; j < b; j++)
						{
							k = j - i;
							SySal.TotalScan.Vertex w = v.Vertices[j];
							a_id[k] = w.Id + 1;
						}
						cmdV.ArrayBindCount = (b - i);
						cmdV.ExecuteNonQuery();
					}
					cmdV.Connection = null;

					if (cmdT == null)
					{
						cmdT = new OracleCommand("INSERT INTO TB_VOLUMETRACKS (ID_EVENTBRICK, ID_RECONSTRUCTION, ID_OPTION, ID, DOWNZ, UPZ, ID_DOWNSTREAMVERTEX, ID_UPSTREAMVERTEX) VALUES (:p_id_eventbrick, :p_id_reconstruction, :p_id_opt, :p_id, :p_downz, :p_upz, NULLIF(:p_iddwvtx, 0), NULLIF(:p_idupvtx, 0))", oracleconn);
						cmdT.Parameters.Add("p_id_eventbrick", OracleDbType.Int64, System.Data.ParameterDirection.Input).Value = a_idbrick;
						cmdT.Parameters.Add("p_id_reconstruction", OracleDbType.Int64, System.Data.ParameterDirection.Input).Value = a_idrec;
						cmdT.Parameters.Add("p_id_opt", OracleDbType.Int64, System.Data.ParameterDirection.Input).Value = a_idopt;
						cmdT.Parameters.Add("p_id", OracleDbType.Int32, System.Data.ParameterDirection.Input).Value = a_id;
						cmdT.Parameters.Add("p_downz", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = a_downz;
						cmdT.Parameters.Add("p_upz", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = a_upz;
						cmdT.Parameters.Add("p_iddwvtx", OracleDbType.Int32, System.Data.ParameterDirection.Input).Value = a_iddwvtx;
						cmdT.Parameters.Add("p_idupvtx", OracleDbType.Int32, System.Data.ParameterDirection.Input).Value = a_idupvtx;
					}
					else cmdT.Connection = oracleconn;
					cmdT.Prepare();

					n = v.Tracks.Length;
					for (i = 0; i < n; i += BatchSize)
					{
						b = i + BatchSize; if (b > n) b = n;
						for (j = i; j < b; j++)
						{
							k = j - i;
							SySal.TotalScan.Track t = v.Tracks[j];
							a_id[k] = t.Id + 1;
							a_downz[k] = t.Downstream_Z;
							a_upz[k] = t.Upstream_Z;
							a_iddwvtx[k] = (t.Downstream_Vertex == null) ? 0 : t.Downstream_Vertex.Id + 1;
							a_idupvtx[k] = (t.Upstream_Vertex == null) ? 0 : t.Upstream_Vertex.Id + 1;
						}
						cmdT.ArrayBindCount = (b - i);
						cmdT.ExecuteNonQuery();
					}
					cmdT.Connection = null;

					if (cmdTS == null)
					{
						cmdTS = new OracleCommand("INSERT INTO TB_VOLTKS_ALIGNMUTKS (ID_EVENTBRICK, ID_RECONSTRUCTION, ID_OPTION, ID_PLATE, ID_ZONE, SIDE, ID_MIPMICROTRACK, ID_VOLUMETRACK) VALUES (:p_id_eventbrick, :p_id_reconstruction, :p_id_opt, :p_id_plate, :p_id_zone, :p_side, :p_id, :p_id_volumetrack)", oracleconn);
						cmdTS.Parameters.Add("p_id_eventbrick", OracleDbType.Int64, System.Data.ParameterDirection.Input).Value = a_idbrick;
						cmdTS.Parameters.Add("p_id_reconstruction", OracleDbType.Int64, System.Data.ParameterDirection.Input).Value = a_idrec;
						cmdTS.Parameters.Add("p_id_opt", OracleDbType.Int64, System.Data.ParameterDirection.Input).Value = a_idopt;
						cmdTS.Parameters.Add("p_id_plate", OracleDbType.Int32, System.Data.ParameterDirection.Input).Value = a_idpl;
						cmdTS.Parameters.Add("p_id_zone", OracleDbType.Int64, System.Data.ParameterDirection.Input).Value = a_idzone;
						cmdTS.Parameters.Add("p_side", OracleDbType.Int32, System.Data.ParameterDirection.Input).Value = a_side;
						cmdTS.Parameters.Add("p_id_mipmicrotrack", OracleDbType.Int32, System.Data.ParameterDirection.Input).Value = a_id;
						cmdTS.Parameters.Add("p_id_volumetrack", OracleDbType.Int32, System.Data.ParameterDirection.Input).Value = a_idvoltk;
					}
					else cmdTS.Connection = oracleconn;
					cmdTS.Prepare();

					n = v.Layers.Length;
					for (h = 0; h < n; h++)
					{
						SySal.TotalScan.Layer lay = v.Layers[h];
						o = lay.Length;
						for (m = i = 0; i < o; i++)
							if (lay[i].TrackOwner != null) m++;						
						for (i = o = 0; i < m; i += BatchSize)
						{
							b = i + BatchSize; if (b > m) b = m;
							for (j = i; j < b; o++)
							{
								k = j - i;
								SySal.TotalScan.Segment s = lay[o];
								if (s.TrackOwner == null) continue;
								DBMIPMicroTrackIndex ix = (DBMIPMicroTrackIndex)(s.Index);
								a_idpl[k] = lay.SheetId;
								a_idzone[k] = ix.ZoneId;
								a_side[k] = lay.Side;
								a_id[k] = ix.Id;
								a_idvoltk[k] = s.TrackOwner.Id + 1;
								j++;
							}
							cmdTS.ArrayBindCount = (b - i);
							cmdTS.ExecuteNonQuery();
						}
					}
					cmdTS.Connection = null;

					if (cmdTA == null)
					{
						cmdTA = new OracleCommand("INSERT INTO TB_VOLUMETRACKS_ATTR (ID_EVENTBRICK, ID_RECONSTRUCTION, ID_OPTION, ID, ID_PROCESSOPERATION, NAME, VALUE) VALUES (:p_id_eventbrick, :p_id_reconstruction, :p_id_opt, :p_id, :p_id_procop, :p_name, :p_value)", oracleconn);
						cmdTA.Parameters.Add("p_id_eventbrick", OracleDbType.Int64, System.Data.ParameterDirection.Input).Value = a_idbrick;
						cmdTA.Parameters.Add("p_id_reconstruction", OracleDbType.Int64, System.Data.ParameterDirection.Input).Value = a_idrec;
						cmdTA.Parameters.Add("p_id_opt", OracleDbType.Int64, System.Data.ParameterDirection.Input).Value = a_idopt;
						cmdTA.Parameters.Add("p_id", OracleDbType.Int32, System.Data.ParameterDirection.Input).Value = a_id;
						cmdTA.Parameters.Add("p_id_procop", OracleDbType.Int64, System.Data.ParameterDirection.Input).Value = a_idproc;
						cmdTA.Parameters.Add("p_name", OracleDbType.Varchar2, System.Data.ParameterDirection.Input).Value = a_name;
						cmdTA.Parameters.Add("p_value", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = a_value;
					}
					else cmdTA.Connection = oracleconn;
					cmdTA.Prepare();

					n = v.Tracks.Length;
					for (h = 0; h < n; h++)
					{
						attrs = v.Tracks[h].ListAttributes();
						m = attrs.Length;
						for (i = 0; i < m; i += BatchSize)
						{
							b = i + BatchSize; if (b > m) b = m;
							for (j = i; j < b; j++)
							{
								k = j - i;
								SySal.TotalScan.Attribute attr = attrs[j];
								SySal.OperaDb.TotalScan.DBNamedAttributeIndex ix = (SySal.OperaDb.TotalScan.DBNamedAttributeIndex)(attr.Index);
								a_id[k] = h + 1;
								a_idproc[k] = ix.ProcOpId;
								a_name[k] = ix.Name;
								a_value[k] = attr.Value;
								j++;
							}
							cmdTA.ArrayBindCount = (b - i);
							cmdTA.ExecuteNonQuery();
						}
					}
					cmdTA.Connection = null;
                    
					if (cmdVA == null)
					{
						cmdVA = new OracleCommand("INSERT INTO TB_VERTICES_ATTR (ID_EVENTBRICK, ID_RECONSTRUCTION, ID_OPTION, ID, ID_PROCESSOPERATION, NAME, VALUE) VALUES (:p_id_eventbrick, :p_id_reconstruction, :p_id_opt, :p_id, :p_id_procop, :p_name, :p_value)", oracleconn);
						cmdVA.Parameters.Add("p_id_eventbrick", OracleDbType.Int64, System.Data.ParameterDirection.Input).Value = a_idbrick;
						cmdVA.Parameters.Add("p_id_reconstruction", OracleDbType.Int64, System.Data.ParameterDirection.Input).Value = a_idrec;
						cmdVA.Parameters.Add("p_id_opt", OracleDbType.Int64, System.Data.ParameterDirection.Input).Value = a_idopt;
						cmdVA.Parameters.Add("p_id", OracleDbType.Int32, System.Data.ParameterDirection.Input).Value = a_id;
						cmdVA.Parameters.Add("p_id_procop", OracleDbType.Int64, System.Data.ParameterDirection.Input).Value = a_idproc;
						cmdVA.Parameters.Add("p_name", OracleDbType.Varchar2, System.Data.ParameterDirection.Input).Value = a_name;
						cmdVA.Parameters.Add("p_value", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = a_value;
					}
					else cmdVA.Connection = oracleconn;
					cmdVA.Prepare();

					n = v.Vertices.Length;
					for (h = 0; h < n; h++)
					{
						attrs = v.Vertices[h].ListAttributes();
						m = attrs.Length;
						for (i = 0; i < m; i += BatchSize)
						{
							b = i + BatchSize; if (b > m) b = m;
							for (j = i; j < b; j++)
							{
								k = j - i;
								SySal.TotalScan.Attribute attr = attrs[j];
								SySal.OperaDb.TotalScan.DBNamedAttributeIndex ix = (SySal.OperaDb.TotalScan.DBNamedAttributeIndex)(attr.Index);
								a_id[k] = h + 1;
								a_idproc[k] = ix.ProcOpId;
								a_name[k] = ix.Name;
								a_value[k] = attr.Value;
								j++;
							}
							cmdVA.ArrayBindCount = (b - i);
							cmdVA.ExecuteNonQuery();
						}
					}
					cmdVA.Connection = null;
                    return id_opt;
				}
			}

		}
	}

	namespace Scanning
	{
		/// <summary>
		/// A minimum ionizing particle track stored in an Opera DB.
		/// </summary>
		public class MIPIndexedEmulsionTrack : SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack
		{
			/// <summary>
			/// Member data on which the DB_Id property relies. Can be accessed by derived classes.
			/// </summary>
			protected long m_DB_Id;
			/// <summary>
			/// The DB identifier of the MIPIndexedEmulsionTrack.
			/// </summary>
			public long DB_Id { get { return m_DB_Id; } }
			/// <summary>
			/// Protected constructor. Prevents users from creating instances of MIPIndexedEmulsionTrack without deriving the class. Is implicitly called by constructors in derived classes.
			/// </summary>
			protected internal MIPIndexedEmulsionTrack() {}

			internal MIPIndexedEmulsionTrack(SySal.Tracking.MIPEmulsionTrackInfo info, int id, long db_id, SySal.Scanning.Plate.IO.OPERA.LinkedZone.View vw)
			{
				m_DB_Id = db_id;
				m_Id = id;
				m_Info = info;
				m_Grains = null;
				m_View = vw;				
			}
		}

		/// <summary>
		/// A base track stored in an Opera DB.
		/// </summary>
		public class MIPBaseTrack : SySal.Scanning.MIPBaseTrack
		{
			/// <summary>
			/// Member data on which the DB_Id property relies. Can be accessed by derived classes.
			/// </summary>
			protected long m_DB_Id;
			/// <summary>
			/// The DB identifier of the MIPBaseTrack.
			/// </summary>
			public long DB_Id { get { return m_DB_Id; } }
			/// <summary>
			/// Protected constructor. Prevents users from creating instances of MIPBaseTrack without deriving the class. Is implicitly called by constructors in derived classes.
			/// </summary>
			protected internal MIPBaseTrack() {}

			internal MIPBaseTrack(SySal.Tracking.MIPEmulsionTrackInfo info, int id, long db_id, SySal.Scanning.MIPIndexedEmulsionTrack toptk, SySal.Scanning.MIPIndexedEmulsionTrack bottomtk)
			{
				m_DB_Id = db_id;	
				m_Id = id;
				m_Info = info;
				m_Top = toptk;
				m_Bottom = bottomtk;
			}
		}

		internal class View : SySal.Scanning.Plate.IO.OPERA.LinkedZone.View
		{
			internal void SetId(int id)
			{
				m_Id = id;
			}

			internal void SetInfo(double px, double py, double tz, double bz)
			{
				m_Position.X = px;
				m_Position.Y = py;
				m_TopZ = tz;
				m_BottomZ = bz;								
			}

			internal void SetSide(SySal.Scanning.Plate.IO.OPERA.LinkedZone.Side side)
			{
				m_Side = side;					
			}

			internal void SetTracks(SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack [] tks)
			{
				m_Tracks = tks;
			}
		}

		internal class Side : SySal.Scanning.Plate.IO.OPERA.LinkedZone.Side
		{
			internal void iSetViews(View [] vw)
			{
				m_Views = vw;
			}

			internal void SetInfo(double tz, double bz)
			{
				m_TopZ = tz;
				m_BottomZ = bz;					
			}

			internal void SetTracks(SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack [] tks)
			{
				m_Tracks = tks;
			}
		}

        /// <summary>
        /// DB procedure calls commonly used in scanning.
        /// </summary>
        public class Procedures
        {
            /// <summary>
            /// Associates a zone to a volume slice.
            /// </summary>
            /// <param name="brickid">the brick being scanned.</param>
            /// <param name="plateid">the plate being scanned.</param>
            /// <param name="volid">the ID_VOLUME of the volume slice.</param>
            /// <param name="zoneid">the ID of the zone - use 0 for NULL.</param>
            /// <param name="damaged">the damage code - use 'N' for no damage.</param>
            /// <param name="conn">the OperaDbConnection to be used.</param>
            /// <param name="trans">the OperaDbTransaction to be used.</param>
            public static void SetVolumeSliceZone(long brickid, int plateid, long volid, long zoneid, char damaged, SySal.OperaDb.OperaDbConnection conn, SySal.OperaDb.OperaDbTransaction trans)
            {
                if (conn.HasBufferTables)
                {
                    new SySal.OperaDb.OperaDbCommand("INSERT INTO OPERA.LZ_SET_VOLUMESLICE_ZONE(P_BRICKID, P_PLATEID, P_VOLID, P_ZONEID, P_DAMAGE) VALUES (" + brickid + ", " + plateid + ", " + volid + ", " + ((zoneid == 0) ? "NULL" : zoneid.ToString()) + ", '" + damaged + "')", conn, trans).ExecuteNonQuery();
                }
                else
                {
                    new SySal.OperaDb.OperaDbCommand("CALL PC_SET_VOLUMESLICE_ZONE(" + brickid + ", " + plateid + ", " + volid + ", " + ((zoneid == 0) ? "NULL" : zoneid.ToString()) + ", '" + damaged + "')", conn, trans).ExecuteNonQuery();
                }
            }

            /// <summary>
            /// Marks a scanback zone as damaged.
            /// </summary>
            /// <param name="brickid">the brick being scanned.</param>
            /// <param name="plateid">the plate being scanned.</param>
            /// <param name="pathid">the scanback path being followed.</param>
            /// <param name="damage">the damaged code ('N' = no damage).</param>
            /// <param name="conn">the OperaDbConnection to be used.</param>
            /// <param name="trans">the OperaDbTransaction to be used.</param>
            public static void ScanbackDamagedZone(long brickid, int plateid, long pathid, char damage, SySal.OperaDb.OperaDbConnection conn, SySal.OperaDb.OperaDbTransaction trans)
            {
                if (conn.HasBufferTables)
                {
                    new SySal.OperaDb.OperaDbCommand("INSERT INTO OPERA.LZ_SCANBACK_DAMAGEDZONE(P_BRICKID, P_PLATEID, P_PATHID, P_DAMAGE) VALUES (" + brickid + ", " + plateid + ", " + pathid + ", '" + damage + "')", conn, trans).ExecuteNonQuery();
                }
                else
                {
                    new SySal.OperaDb.OperaDbCommand("CALL PC_SCANBACK_DAMAGEDZONE(" + brickid + ", " + plateid + ", " + pathid + ", '" + damage + "')", conn, trans).ExecuteNonQuery();
                }
            }

            /// <summary>
            /// Defines a zone as correctly scanned, without finding a candidate.
            /// </summary>
            /// <param name="brickid">the brick being scanned.</param>
            /// <param name="plateid">the plate being scanned.</param>
            /// <param name="pathid">the scanback path being followed.</param>
            /// <param name="zoneid">the id of the zone scanned.</param>
            /// <param name="conn">the OperaDbConnection to be used.</param>
            /// <param name="trans">the OperaDbTransaction to be used.</param>
            public static void ScanbackNoCandidate(long brickid, int plateid, long pathid, long zoneid, SySal.OperaDb.OperaDbConnection conn, SySal.OperaDb.OperaDbTransaction trans)
            {
                if (conn.HasBufferTables)
                {
                    new SySal.OperaDb.OperaDbCommand("INSERT INTO OPERA.LZ_SCANBACK_NOCANDIDATE(P_BRICKID, P_PLATEID, P_PATHID, P_ZONEID) VALUES (" + brickid + ", " + plateid + ", " + pathid + ", " + zoneid + ")", conn, trans).ExecuteNonQuery();
                }
                else
                {
                    new SySal.OperaDb.OperaDbCommand("CALL PC_SCANBACK_NOCANDIDATE(" + brickid + ", " + plateid + ", " + pathid + ", " + zoneid + ")", conn, trans).ExecuteNonQuery();
                }
            }

            /// <summary>
            /// Defines a zone as correctly scanned, adding the link to the found candidate.
            /// </summary>
            /// <param name="brickid">the brick being scanned.</param>
            /// <param name="plateid">the plate being scanned.</param>
            /// <param name="pathid">the scanback path being followed.</param>
            /// <param name="zoneid">the id of the zone scanned.</param>
            /// <param name="candid">the id of the base track identified as the candidate.</param>
            /// <param name="conn">the OperaDbConnection to be used.</param>
            /// <param name="trans">the OperaDbTransaction to be used.</param>
            public static void ScanbackCandidate(long brickid, int plateid, long pathid, long zoneid, long candid, bool ismanual, SySal.OperaDb.OperaDbConnection conn, SySal.OperaDb.OperaDbTransaction trans)
            {
                if (conn.HasBufferTables)
                {
                    new SySal.OperaDb.OperaDbCommand("INSERT INTO OPERA.LZ_SCANBACK_CANDIDATE(P_BRICKID, P_PLATEID, P_PATHID, P_ZONEID, P_CANDID, P_MANUAL) VALUES (" + brickid + ", " + plateid + ", " + pathid + ", " + zoneid + ", " + candid + ", " + (ismanual ? "1" : "0") + ")", conn, trans).ExecuteNonQuery();
                }
                else
                {
                    new SySal.OperaDb.OperaDbCommand("CALL PC_SCANBACK_CANDIDATE(" + brickid + ", " + plateid + ", " + pathid + ", " + zoneid + "," + candid + ", " + (ismanual ? "1" : "0") + ")", conn, trans).ExecuteNonQuery();
                }
            }

            /// <summary>
            /// Forks a scanback path.
            /// </summary>
            /// <param name="brickid">the brick being scanned.</param>
            /// <param name="plateid">the plate being scanned.</param>
            /// <param name="pathid">the scanback path being followed, from which the new path originates.</param>
            /// <param name="zoneid">the id of the zone scanned.</param>
            /// <param name="candid">the id of the base track identified as the candidate.</param>
            /// <param name="forkid">the value of PATH for the new path to be created.</param>
            /// <param name="conn">the OperaDbConnection to be used.</param>
            /// <param name="trans">the OperaDbTransaction to be used.</param>
            public static void ScanbackFork(long brickid, int plateid, long pathid, long zoneid, long candid, long forkid, SySal.OperaDb.OperaDbConnection conn, SySal.OperaDb.OperaDbTransaction trans)
            {
                if (conn.HasBufferTables)
                {
                    new SySal.OperaDb.OperaDbCommand("INSERT INTO OPERA.LZ_SCANBACK_FORK(P_BRICKID, P_PLATEID, P_PATHID, P_ZONEID, P_CANDID, P_FORKID) VALUES (" + brickid + ", " + plateid + ", " + pathid + ", " + zoneid + ", " + candid + ", " + forkid + ")", conn, trans).ExecuteNonQuery();
                }
                else
                {
                    new SySal.OperaDb.OperaDbCommand("CALL PC_SCANBACK_FORK(" + brickid + ", " + plateid + ", " + pathid + ", " + zoneid + "," + candid + ", " + forkid + ")", conn, trans).ExecuteNonQuery();
                }
            }

            /// <summary>
            /// Cancels a scanback path.
            /// </summary>
            /// <param name="brickid">the brick being scanned.</param>
            /// <param name="pathid">the path to be cancelled.</param>
            /// <param name="pathid">the scanback path being followed.</param>
            /// <param name="conn">the OperaDbConnection to be used.</param>
            /// <param name="trans">the OperaDbTransaction to be used.</param>
            public static void ScanbackCancelPath(long brickid, long pathid, int plateid, SySal.OperaDb.OperaDbConnection conn, SySal.OperaDb.OperaDbTransaction trans)
            {
                if (conn.HasBufferTables)
                {
                    new SySal.OperaDb.OperaDbCommand("INSERT INTO OPERA.LZ_SCANBACK_CANCEL_PATH(P_BRICKID, P_PATHID, PLATEID) VALUES (" + brickid + ", " + pathid + ", " + plateid + ")", conn, trans).ExecuteNonQuery();
                }
                else
                {
                    new SySal.OperaDb.OperaDbCommand("CALL PC_SCANBACK_CANCEL_PATH(" + brickid + ", " + pathid + ", " + plateid + ")", conn, trans).ExecuteNonQuery();
                }
            }

        }

		/// <summary>
		/// An Opera LinkedZone stored in an Opera DB.
		/// </summary>
		public class LinkedZone : SySal.Scanning.Plate.IO.OPERA.LinkedZone
		{
			/// <summary>
			/// Member data on which the DB_Brick_Id property relies. Can be accessed by derived classes.
			/// </summary>
			protected long m_DB_Brick_Id;
			/// <summary>
			/// The DB identifier of the brick to which the zone belongs.
			/// </summary>
			public long DB_Brick_Id { get { return m_DB_Brick_Id; } }
			/// <summary>
			/// Member data on which the DB_Plate_Id property relies. Can be accessed by derived classes.
			/// </summary>
			protected long m_DB_Plate_Id;
			/// <summary>
			/// The DB identifier of the plate to which the zone belongs.
			/// </summary>
			public long DB_Plate_Id { get { return m_DB_Plate_Id; } }
			/// <summary>
			/// Member data on which the DB_Id property relies. Can be accessed by derived classes.
			/// </summary>
			protected long m_DB_Id;
			/// <summary>
			/// The DB identifier of the LinkedZone.
			/// </summary>
			public long DB_Id { get { return m_DB_Id; } }
			/// <summary>
			/// Member data on which the DB_ProcessOperation_Id property relies. Can be accessed by derived classes.
			/// </summary>
			protected long m_DB_ProcessOperation_Id;
			/// <summary>
			/// The DB identifier of the process operation to which the LinkedZone belongs.
			/// </summary>
			public long DB_ProcessOperation_Id { get { return m_DB_ProcessOperation_Id; } }
			/// <summary>
			/// Member data on which the Series property relies. Can be accessed by derived classes.
			/// </summary>			
			protected long m_Series;
			/// <summary>
			/// The serial number of the LinkedZone.
			/// </summary>
			public long Series { get { return m_Series; } }
			/// <summary>
			/// Member data on which the RawDataPath property relies. Can be accessed by derived classes.
			/// </summary>
			protected string m_RawDataPath;
			/// <summary>
			/// Path of the raw data files from which this scanning zone originates.
			/// </summary>
			public string RawDataPath { get { return m_RawDataPath; } }
			/// <summary>
			/// Member data on which the StartTime property relies. Can be accessed by derived classes.
			/// </summary>
			protected DateTime m_StartTime;
			/// <summary>
			/// Date and time when the scanning of this zone started.
			/// </summary>
			public DateTime StartTime { get { return m_StartTime; } }
			/// <summary>
			/// Member data on which the EndTime property relies. Can be accessed by derived classes.
			/// </summary>
			protected DateTime m_EndTime;
			/// <summary>
			/// Date and time when the scanning of this zone ended.
			/// </summary>
			public DateTime EndTime { get { return m_EndTime; } }

			/// <summary>
			/// Protected constructor. Prevents users from creating instances of LinkedZone without deriving the class. Is implicitly called by constructors in derived classes.
			/// </summary>
			protected LinkedZone() {}

			/// <summary>
			/// The detail level desired when restoring LinkedZones from the DB.
			/// </summary>
			public enum DetailLevel
			{
				/// <summary>
				/// Restore all information, including relationships between base tracks and microtracks.
				/// </summary>
				Full = 0,
				/// <summary>
				/// Restore geometrical and quality parameters of base tracks.
				/// </summary>
				BaseFull = -1,
				/// <summary>
				/// Restore only geometrical parameters of base tracks.
				/// </summary>
				BaseGeom = -2
			}

			/// <summary>
			/// Reads a LinkedZone from the OperaDb, using the specified connection, transaction, and the DB identifier for the zone to be read.
			/// </summary>
			/// <param name="conn">the DB connection to be used.</param>
			/// <param name="trans">the transaction to be used. Can be null.</param>
			/// <param name="db_brick_id">the DB identifier of the brick to which the zone belongs.</param>
			/// <param name="db_id">the DB identifier of the zone to be read.</param>
			/// <param name="detail">the detail level required for the zone information.</param>
			public LinkedZone(OperaDbConnection conn, OperaDbTransaction trans, long db_brick_id, long db_id, DetailLevel detail)
			{
				OracleConnection oracleconn = (OracleConnection)conn.Conn;				

				switch (conn.DBSchemaVersion)
				{
					case DBSchemaVersion.Basic_V1:
					{
#region basic_v1_lz_read
						DataSet ds = new DataSet();
						OracleDataAdapter da;
						da = new OracleDataAdapter("SELECT /*+INDEX(TB_ZONES PK_ZONES) */ ID_PLATE, ID_PROCESSOPERATION, MINX, MAXX, MINY, MAXY, RAWDATAPATH, STARTTIME, ENDTIME, BASETHICKNESS, SERIES FROM TB_ZONES WHERE(TB_ZONES.ID_EVENTBRICK = " + db_brick_id + " AND TB_ZONES.ID = " + db_id + ")", oracleconn);
						da.Fill(ds);
						DataTable dt = ds.Tables[0];
						DataRow drz = dt.Rows[0];
						m_DB_Brick_Id = db_brick_id;
						m_DB_Id = db_id;
						m_DB_Plate_Id = Convert.ToInt64(drz[0]);
						m_DB_ProcessOperation_Id = Convert.ToInt64(drz[1]);
						m_Extents.MinX = Convert.ToDouble(drz[2]);
						m_Extents.MaxX = Convert.ToDouble(drz[3]);
						m_Extents.MinY = Convert.ToDouble(drz[4]);
						m_Extents.MaxY = Convert.ToDouble(drz[5]);
						m_RawDataPath = drz[6].ToString();
						m_StartTime = (DateTime)drz[7];
						m_EndTime = (DateTime)drz[8];
						double thickness = Convert.ToDouble(drz[9]);
						m_Series = Convert.ToInt64(drz[10]);

						m_Id.Part0 = (int)m_DB_Brick_Id;
						m_Id.Part1 = (int)m_DB_Plate_Id;
						m_Id.Part2 = (int)m_Series;
						m_Id.Part3 = 0;
						m_Center.X = 0.5 * (m_Extents.MinX + m_Extents.MaxX);
						m_Center.Y = 0.5 * (m_Extents.MinY + m_Extents.MaxY);
						m_Transform = new SySal.DAQSystem.Scanning.IntercalibrationInfo();
						m_Transform.MXX = m_Transform.MYY = 1.0;
						m_Transform.MXY = m_Transform.MYX = 0.0;
						m_Transform.TX = m_Transform.TY = 0.0;
						m_Transform.RX = m_Transform.RY = 0.0;

						if (detail == DetailLevel.BaseGeom)
						{
							SySal.OperaDb.Scanning.MIPIndexedEmulsionTrack zerotrack = new SySal.OperaDb.Scanning.MIPIndexedEmulsionTrack(new SySal.Tracking.MIPEmulsionTrackInfo(), 0, 0, null);
							ds = new DataSet();
							da = new OracleDataAdapter("SELECT /*+INDEX(TB_MIPBASETRACKS PK_MIPBASETRACKS) */ ID, POSX, POSY, SLOPEX, SLOPEY FROM TB_MIPBASETRACKS WHERE(TB_MIPBASETRACKS.ID_EVENTBRICK = " + db_brick_id + " AND TB_MIPBASETRACKS.ID_ZONE = " + db_id + ")", oracleconn);
							da.Fill(ds);
							dt = ds.Tables[0];
							m_Tracks = new SySal.OperaDb.Scanning.MIPBaseTrack[dt.Rows.Count];
							foreach (DataRow dr in dt.Rows)
							{
								SySal.Tracking.MIPEmulsionTrackInfo info = new SySal.Tracking.MIPEmulsionTrackInfo();
								info.Intercept.X = Convert.ToDouble(dr[1]);
								info.Intercept.Y = Convert.ToDouble(dr[2]);
								info.Slope.X = Convert.ToDouble(dr[3]);
								info.Slope.Y = Convert.ToDouble(dr[4]);
								info.Slope.Z = 1.0;
								info.TopZ = 43.0;
								info.BottomZ = -thickness - info.BottomZ;
								int id = Convert.ToInt32(dr[0]);
								m_Tracks[id - 1] = new SySal.OperaDb.Scanning.MIPBaseTrack(info, (int)id - 1, (int)id, zerotrack, zerotrack);
							}

							SySal.OperaDb.Scanning.Side topside = new SySal.OperaDb.Scanning.Side();
							m_Top = topside;
							SySal.OperaDb.Scanning.View topview = new SySal.OperaDb.Scanning.View();
							topview.SetSide(topside);
							topview.SetId(0);
							topview.SetInfo(m_Center.X, m_Center.Y, 43.0, 0.0);
							topview.SetTracks(new SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack[1] {zerotrack});
							topside.SetInfo(43.0, 0.0);
							topside.iSetViews(new SySal.OperaDb.Scanning.View[1] { topview });
							topside.SetTracks(new SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack[1] {zerotrack});
							
							SySal.OperaDb.Scanning.Side bottomside = new SySal.OperaDb.Scanning.Side();
							m_Bottom = bottomside;
							SySal.OperaDb.Scanning.View bottomview = new SySal.OperaDb.Scanning.View();
							bottomview.SetSide(bottomside);
							bottomview.SetId(0);
							bottomview.SetInfo(m_Center.X, m_Center.Y, -thickness, -thickness-43.0);
							bottomview.SetTracks(new SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack[1] {zerotrack});
							bottomside.SetInfo(-thickness, -thickness-43.0);
							bottomside.iSetViews(new SySal.OperaDb.Scanning.View[1] { bottomview });
							bottomside.SetTracks(new SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack[1] {zerotrack});

						}
						else if (detail == DetailLevel.BaseFull)
						{
							SySal.OperaDb.Scanning.MIPIndexedEmulsionTrack zerotrack = new SySal.OperaDb.Scanning.MIPIndexedEmulsionTrack(new SySal.Tracking.MIPEmulsionTrackInfo(), 0, 0, null);
							ds = new DataSet();
							da = new OracleDataAdapter("SELECT /*+INDEX(TB_MIPBASETRACKS PK_MIPBASETRACKS) */ ID, POSX, POSY, SLOPEX, SLOPEY, NVL(GRAINS, FLOOR(PH * 0.001)), NVL(AREASUM, MOD(PH, 1000)), SIGMA FROM TB_MIPBASETRACKS WHERE(TB_MIPBASETRACKS.ID_EVENTBRICK = " + db_brick_id + " AND TB_MIPBASETRACKS.ID_ZONE = " + db_id + ")", oracleconn);
							da.Fill(ds);
							dt = ds.Tables[0];
							m_Tracks = new SySal.OperaDb.Scanning.MIPBaseTrack[dt.Rows.Count];
							foreach (DataRow dr in dt.Rows)
							{
								SySal.Tracking.MIPEmulsionTrackInfo info = new SySal.Tracking.MIPEmulsionTrackInfo();
								info.Intercept.X = Convert.ToDouble(dr[1]);
								info.Intercept.Y = Convert.ToDouble(dr[2]);
								info.Slope.X = Convert.ToDouble(dr[3]);
								info.Slope.Y = Convert.ToDouble(dr[4]);
								info.Slope.Z = 1.0;
								info.Count = Convert.ToUInt16(dr[5]);
								info.AreaSum = Convert.ToUInt32(dr[6]);
								info.Sigma = Convert.ToDouble(dr[7]);
								info.TopZ = 43.0;
								info.BottomZ = -thickness - info.BottomZ;
								int id = Convert.ToInt32(dr[0]);
								m_Tracks[id - 1] = new SySal.OperaDb.Scanning.MIPBaseTrack(info, (int)id - 1, (int)id, zerotrack, zerotrack);
							}

							SySal.OperaDb.Scanning.Side topside = new SySal.OperaDb.Scanning.Side();
							m_Top = topside;
							SySal.OperaDb.Scanning.View topview = new SySal.OperaDb.Scanning.View();
							topview.SetSide(topside);
							topview.SetId(0);
							topview.SetInfo(m_Center.X, m_Center.Y, 43.0, 0.0);
							topview.SetTracks(new SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack[1] {zerotrack});
							topside.SetInfo(43.0, 0.0);
							topside.iSetViews(new SySal.OperaDb.Scanning.View[1] { topview });
							topside.SetTracks(new SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack[1] {zerotrack});
							
							SySal.OperaDb.Scanning.Side bottomside = new SySal.OperaDb.Scanning.Side();
							m_Bottom = bottomside;
							SySal.OperaDb.Scanning.View bottomview = new SySal.OperaDb.Scanning.View();
							bottomview.SetSide(bottomside);
							bottomview.SetId(0);
							bottomview.SetInfo(m_Center.X, m_Center.Y, -thickness, -thickness-43.0);
							bottomview.SetTracks(new SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack[1] {zerotrack});
							bottomside.SetInfo(-thickness, -thickness-43.0);
							bottomside.iSetViews(new SySal.OperaDb.Scanning.View[1] { bottomview });
							bottomside.SetTracks(new SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack[1] {zerotrack});
						}
						else
						{
							SySal.OperaDb.Scanning.View topview = new SySal.OperaDb.Scanning.View();
							SySal.OperaDb.Scanning.View bottomview = new SySal.OperaDb.Scanning.View();

							ds = new DataSet();
							da = new OracleDataAdapter("SELECT /*+INDEX(TB_MIPMICROTRACKS PK_MIPMICROTRACKS) */ ID, POSX, POSY, SLOPEX, SLOPEY, NVL(GRAINS, FLOOR(PH * 0.001)), NVL(AREASUM, MOD(PH, 1000)), SIGMA, SIDE FROM TB_MIPMICROTRACKS WHERE(TB_MIPMICROTRACKS.ID_EVENTBRICK = " + db_brick_id + " AND TB_MIPMICROTRACKS.ID_ZONE = " + db_id + ")", oracleconn);
							da.Fill(ds);
							dt = ds.Tables[0];
							int toptkscount = 0;
							foreach (DataRow dr in dt.Rows)
								if (Convert.ToInt32(dr[8]) == 1) toptkscount++;
							SySal.OperaDb.Scanning.MIPIndexedEmulsionTrack [] toptks = new SySal.OperaDb.Scanning.MIPIndexedEmulsionTrack [toptkscount];
							SySal.OperaDb.Scanning.MIPIndexedEmulsionTrack [] bottomtks = new SySal.OperaDb.Scanning.MIPIndexedEmulsionTrack [dt.Rows.Count - toptkscount];
							foreach (DataRow dr in dt.Rows)
							{
								SySal.Tracking.MIPEmulsionTrackInfo info = new SySal.Tracking.MIPEmulsionTrackInfo();
								int id = Convert.ToInt32(dr[0]);
								info.Intercept.X = Convert.ToDouble(dr[1]);
								info.Intercept.Y = Convert.ToDouble(dr[2]);
								info.Slope.X = Convert.ToDouble(dr[3]);
								info.Slope.Y = Convert.ToDouble(dr[4]);
								info.Slope.Z = 1.0;
								info.Count = Convert.ToUInt16(dr[5]);
								info.AreaSum = Convert.ToUInt32(dr[6]);
								info.Sigma = Convert.ToDouble(dr[7]);
								if (Convert.ToInt32(dr[8]) == 2) 
								{
									info.TopZ = -thickness;
									info.BottomZ = -thickness - 43.0;
									bottomtks[id - 1] = new SySal.OperaDb.Scanning.MIPIndexedEmulsionTrack(info, id - 1, id, bottomview);
								}
								else
								{
									info.TopZ = 43.0;
									info.BottomZ = 0.0;
									toptks[id - 1] = new SySal.OperaDb.Scanning.MIPIndexedEmulsionTrack(info, id - 1, id, topview);
								}
							}

							SySal.OperaDb.Scanning.Side topside = new SySal.OperaDb.Scanning.Side();
							m_Top = topside;
							topview.SetSide(topside);
							topview.SetId(0);
							topview.SetInfo(m_Center.X, m_Center.Y, 43.0, 0.0);
							topview.SetTracks(toptks);
							topside.SetInfo(43.0, 0.0);
							topside.iSetViews(new SySal.OperaDb.Scanning.View[1] { topview });
							topside.SetTracks(toptks);
							
							SySal.OperaDb.Scanning.Side bottomside = new SySal.OperaDb.Scanning.Side();
							m_Bottom = bottomside;
							bottomview.SetSide(bottomside);
							bottomview.SetId(0);
							bottomview.SetInfo(m_Center.X, m_Center.Y, -thickness, -thickness-43.0);
							bottomview.SetTracks(bottomtks);
							bottomside.SetInfo(-thickness, -thickness-43.0);
							bottomside.iSetViews(new SySal.OperaDb.Scanning.View[1] { bottomview });
							bottomside.SetTracks(bottomtks);

							ds = new DataSet();
							da = new OracleDataAdapter("SELECT /*+INDEX(TB_MIPBASETRACKS PK_MIPBASETRACKS) */ ID, POSX, POSY, SLOPEX, SLOPEY, NVL(GRAINS, FLOOR(PH * 0.001)), NVL(AREASUM, MOD(PH, 1000)), SIGMA, ID_DOWNTRACK, ID_UPTRACK FROM TB_MIPBASETRACKS WHERE(TB_MIPBASETRACKS.ID_EVENTBRICK = " + db_brick_id + " AND TB_MIPBASETRACKS.ID_ZONE = " + db_id + ")", oracleconn);
							da.Fill(ds);
							dt = ds.Tables[0];
							m_Tracks = new SySal.OperaDb.Scanning.MIPBaseTrack[dt.Rows.Count];
							foreach (DataRow dr in dt.Rows)
							{
								SySal.Tracking.MIPEmulsionTrackInfo info = new SySal.Tracking.MIPEmulsionTrackInfo();
								info.Intercept.X = Convert.ToDouble(dr[1]);
								info.Intercept.Y = Convert.ToDouble(dr[2]);
								info.Slope.X = Convert.ToDouble(dr[3]);
								info.Slope.Y = Convert.ToDouble(dr[4]);
								info.Slope.Z = 1.0;
								info.Count = Convert.ToUInt16(dr[5]);
								info.AreaSum = Convert.ToUInt32(dr[6]);
								info.Sigma = Convert.ToDouble(dr[7]);
								info.TopZ = 43.0;
								info.BottomZ = -thickness - info.BottomZ;
								int id = Convert.ToInt32(dr[0]);
								m_Tracks[id - 1] = new SySal.OperaDb.Scanning.MIPBaseTrack(info, (int)id - 1, (int)id, toptks[Convert.ToInt64(dr[8]) - 1], bottomtks[Convert.ToInt64(dr[9]) - 1]);
							}
						}						
#endregion
					}
						break;

					case DBSchemaVersion.HasViews_V2:
					{
#region hasviews_v2_lz_read
						DataSet ds = new DataSet();
						OracleDataAdapter da;
						da = new OracleDataAdapter("SELECT /*+INDEX(TB_ZONES PK_ZONES) */ ID_PLATE, ID_PROCESSOPERATION, MINX, MAXX, MINY, MAXY, RAWDATAPATH, STARTTIME, ENDTIME, SERIES, TXX, TXY, TYX, TYY, TDX, TDY FROM TB_ZONES WHERE(TB_ZONES.ID_EVENTBRICK = " + db_brick_id + " AND TB_ZONES.ID = " + db_id + ")", oracleconn);
						da.Fill(ds);
						DataTable dt = ds.Tables[0];
						DataRow drz = dt.Rows[0];
						m_DB_Brick_Id = db_brick_id;
						m_DB_Id = db_id;
						m_DB_Plate_Id = Convert.ToInt64(drz[0]);
						m_DB_ProcessOperation_Id = Convert.ToInt64(drz[1]);
						m_Extents.MinX = Convert.ToDouble(drz[2]);
						m_Extents.MaxX = Convert.ToDouble(drz[3]);
						m_Extents.MinY = Convert.ToDouble(drz[4]);
						m_Extents.MaxY = Convert.ToDouble(drz[5]);
						m_RawDataPath = drz[6].ToString();
						m_StartTime = (DateTime)drz[7];
						m_EndTime = (DateTime)drz[8];
						m_Series = Convert.ToInt64(drz[9]);
						m_Transform = new SySal.DAQSystem.Scanning.IntercalibrationInfo();
						m_Transform.MXX = Convert.ToDouble(drz[10]);
						m_Transform.MXY = Convert.ToDouble(drz[11]);
						m_Transform.MYX = Convert.ToDouble(drz[12]);
						m_Transform.MYY = Convert.ToDouble(drz[13]);
						m_Transform.TX = Convert.ToDouble(drz[14]);
						m_Transform.TY = Convert.ToDouble(drz[15]);
						m_Transform.TZ = m_Transform.RX = m_Transform.RY = 0.0;

						m_Id.Part0 = (int)m_DB_Brick_Id;
						m_Id.Part1 = (int)m_DB_Plate_Id;
						m_Id.Part2 = (int)m_Series;
						m_Id.Part3 = 0;
						m_Center.X = 0.5 * (m_Extents.MinX + m_Extents.MaxX);
						m_Center.Y = 0.5 * (m_Extents.MinY + m_Extents.MaxY);


						DataSet dv = new DataSet();
						new OracleDataAdapter("SELECT /*+INDEX(TB_VIEWS PK_VIEWS) */ SIDE, ID, DOWNZ, UPZ, POSX, POSY FROM TB_VIEWS WHERE(TB_VIEWS.ID_EVENTBRICK = " + db_brick_id + " AND TB_VIEWS.ID_ZONE = " + db_id + ") ORDER BY SIDE, ID", oracleconn).Fill(dv);
						DataRowCollection dtv = dv.Tables[0].Rows;
						int topvwcount = 0, bottomvwcount = 0;
						for (topvwcount = 0; topvwcount < dtv.Count && Convert.ToInt32(dtv[topvwcount][0]) == 1; topvwcount++);
						for (bottomvwcount = topvwcount; bottomvwcount < dtv.Count && Convert.ToInt32(dtv[bottomvwcount][0]) == 2; bottomvwcount++);
						bottomvwcount -= topvwcount;
						SySal.OperaDb.Scanning.View [] topviews = new SySal.OperaDb.Scanning.View[topvwcount];
						SySal.OperaDb.Scanning.View [] bottomviews = new SySal.OperaDb.Scanning.View[bottomvwcount];
						double topext = 0.0, topint = 0.0, bottomint = 0.0, bottomext = 0.0;
						foreach (System.Data.DataRow dr in dtv)
						{
							SySal.OperaDb.Scanning.View vw = new SySal.OperaDb.Scanning.View();
							vw.SetId(Convert.ToInt32(dr[1]) - 1);
							//if (vw.Id != topvwcount) throw new Exception("Inconsistency found in view sequence, top side, view #" + vw.Id);
							vw.SetInfo(Convert.ToDouble(dr[4]), Convert.ToDouble(dr[5]), Convert.ToDouble(dr[2]), Convert.ToDouble(dr[3]));							
							if (Convert.ToInt32(dr[0]) == 1)
							{
								topviews[vw.Id] = vw;
								topext += vw.TopZ;
								topint += vw.BottomZ;
							}
							else
							{
								bottomviews[vw.Id] = vw;
								bottomint += vw.TopZ;
								bottomext += vw.BottomZ;
							}
						}
						if (topvwcount == 0 && bottomvwcount == 0)
						{
							topext = 43.0;
							topint = 0.0;
							bottomint = -210.0;
							bottomext = -253.0;
						}
						else
						{
							if (topvwcount > 0)
							{
								topext /= topvwcount;
								topint /= topvwcount;
								if (bottomvwcount == 0)
								{
									bottomint = topint - 210.0;
									bottomext = bottomint - 43.0;
								}
							}
							if (bottomvwcount > 0)
							{
								bottomext /= topvwcount;
								bottomint /= topvwcount;
								if (topvwcount == 0)
								{
									topint = bottomint + 210.0;
									topext = topint + 43.0;
								}
							}
						}
						SySal.OperaDb.Scanning.Side topside = new SySal.OperaDb.Scanning.Side();
						topside.SetInfo(topext, topint);
						topside.iSetViews(topviews);
						m_Top = topside;
						for (topvwcount = 0; topvwcount < topviews.Length; topvwcount++)
							((SySal.OperaDb.Scanning.View)(topviews[topvwcount])).SetSide(topside);
						SySal.OperaDb.Scanning.Side bottomside = new SySal.OperaDb.Scanning.Side();
						bottomside.SetInfo(bottomint, bottomext);
						bottomside.iSetViews(bottomviews);
						m_Bottom = bottomside;
						for (bottomvwcount = 0; bottomvwcount < bottomviews.Length; bottomvwcount++)
							((SySal.OperaDb.Scanning.View)(bottomviews[bottomvwcount])).SetSide(bottomside);

						if (detail == DetailLevel.BaseGeom)
						{
							SySal.OperaDb.Scanning.MIPIndexedEmulsionTrack zerotrack = new SySal.OperaDb.Scanning.MIPIndexedEmulsionTrack(new SySal.Tracking.MIPEmulsionTrackInfo(), 0, 0, null);
							ds = new DataSet();
							da = new OracleDataAdapter("SELECT /*+INDEX(TB_MIPBASETRACKS PK_MIPBASETRACKS) */ ID, POSX, POSY, SLOPEX, SLOPEY FROM TB_MIPBASETRACKS WHERE(TB_MIPBASETRACKS.ID_EVENTBRICK = " + db_brick_id + " AND TB_MIPBASETRACKS.ID_ZONE = " + db_id + ")", oracleconn);
							da.Fill(ds);
							dt = ds.Tables[0];
							m_Tracks = new SySal.OperaDb.Scanning.MIPBaseTrack[dt.Rows.Count];
							foreach (DataRow dr in dt.Rows)
							{
								SySal.Tracking.MIPEmulsionTrackInfo info = new SySal.Tracking.MIPEmulsionTrackInfo();
								info.Intercept.X = Convert.ToDouble(dr[1]);
								info.Intercept.Y = Convert.ToDouble(dr[2]);
								info.Intercept.Z = topint;
								info.Slope.X = Convert.ToDouble(dr[3]);
								info.Slope.Y = Convert.ToDouble(dr[4]);
								info.Slope.Z = 1.0;
								info.TopZ = topext;
								info.BottomZ = bottomext;
								int id = Convert.ToInt32(dr[0]);
								m_Tracks[id - 1] = new SySal.OperaDb.Scanning.MIPBaseTrack(info, (int)id - 1, (int)id, zerotrack, zerotrack);
							}
							topside.SetTracks(new SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack[1] { zerotrack } );
							bottomside.SetTracks(new SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack[1] { zerotrack } );
						}
						else if (detail == DetailLevel.BaseFull)
						{
							SySal.OperaDb.Scanning.MIPIndexedEmulsionTrack zerotrack = new SySal.OperaDb.Scanning.MIPIndexedEmulsionTrack(new SySal.Tracking.MIPEmulsionTrackInfo(), 0, 0, null);
							ds = new DataSet();
							da = new OracleDataAdapter("SELECT /*+INDEX(TB_MIPBASETRACKS PK_MIPBASETRACKS) */ ID, POSX, POSY, SLOPEX, SLOPEY, NVL(GRAINS, FLOOR(PH * 0.001)), NVL(AREASUM, MOD(PH, 1000)), SIGMA FROM TB_MIPBASETRACKS WHERE(TB_MIPBASETRACKS.ID_EVENTBRICK = " + db_brick_id + " AND TB_MIPBASETRACKS.ID_ZONE = " + db_id + ")", oracleconn);
							da.Fill(ds);
							dt = ds.Tables[0];
							m_Tracks = new SySal.OperaDb.Scanning.MIPBaseTrack[dt.Rows.Count];
							foreach (DataRow dr in dt.Rows)
							{
								SySal.Tracking.MIPEmulsionTrackInfo info = new SySal.Tracking.MIPEmulsionTrackInfo();
								info.Intercept.X = Convert.ToDouble(dr[1]);
								info.Intercept.Y = Convert.ToDouble(dr[2]);
								info.Intercept.Z = topint;
								info.Slope.X = Convert.ToDouble(dr[3]);
								info.Slope.Y = Convert.ToDouble(dr[4]);
								info.Slope.Z = 1.0;
								info.Count = Convert.ToUInt16(dr[5]);
								info.AreaSum = Convert.ToUInt32(dr[6]);
								info.Sigma = Convert.ToDouble(dr[7]);
								info.TopZ = topext;
								info.BottomZ = bottomext;
								int id = Convert.ToInt32(dr[0]);
								m_Tracks[id - 1] = new SySal.OperaDb.Scanning.MIPBaseTrack(info, (int)id - 1, (int)id, zerotrack, zerotrack);
							}
							topside.SetTracks(new SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack[1] { zerotrack } );
							bottomside.SetTracks(new SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack[1] { zerotrack } );
						}
						else
						{
							System.Collections.ArrayList [] toplists = new System.Collections.ArrayList[topviews.Length];
							System.Collections.ArrayList [] bottomlists = new System.Collections.ArrayList[bottomviews.Length];
							int i;
							for (i = 0; i < toplists.Length; i++)
								toplists[i] = new System.Collections.ArrayList();
							for (i = 0; i < bottomlists.Length; i++)
								bottomlists[i] = new System.Collections.ArrayList();

							ds = new DataSet();
							da = new OracleDataAdapter("SELECT /*+INDEX(TB_MIPMICROTRACKS PK_MIPMICROTRACKS) */ ID, POSX, POSY, SLOPEX, SLOPEY, NVL(GRAINS, FLOOR(PH * 0.001)), NVL(AREASUM, MOD(PH, 1000)), SIGMA, SIDE, ID_VIEW FROM TB_MIPMICROTRACKS WHERE(TB_MIPMICROTRACKS.ID_EVENTBRICK = " + db_brick_id + " AND TB_MIPMICROTRACKS.ID_ZONE = " + db_id + ")", oracleconn);
							da.Fill(ds);
							dt = ds.Tables[0];
							int toptkscount = 0;
							foreach (DataRow dr in dt.Rows)
								if (Convert.ToInt32(dr[8]) == 1) toptkscount++;
							SySal.OperaDb.Scanning.MIPIndexedEmulsionTrack [] toptks = new SySal.OperaDb.Scanning.MIPIndexedEmulsionTrack [toptkscount];
							SySal.OperaDb.Scanning.MIPIndexedEmulsionTrack [] bottomtks = new SySal.OperaDb.Scanning.MIPIndexedEmulsionTrack [dt.Rows.Count - toptkscount];
							foreach (DataRow dr in dt.Rows)
							{
								SySal.Tracking.MIPEmulsionTrackInfo info = new SySal.Tracking.MIPEmulsionTrackInfo();
								int id = Convert.ToInt32(dr[0]);
								info.Intercept.X = Convert.ToDouble(dr[1]);
								info.Intercept.Y = Convert.ToDouble(dr[2]);								
								info.Slope.X = Convert.ToDouble(dr[3]);
								info.Slope.Y = Convert.ToDouble(dr[4]);
								info.Slope.Z = 1.0;
								info.Count = Convert.ToUInt16(dr[5]);
								info.AreaSum = Convert.ToUInt32(dr[6]);
								info.Sigma = Convert.ToDouble(dr[7]);
								if (Convert.ToInt32(dr[8]) == 2) 
								{
									View vw = bottomviews[Convert.ToInt32(dr[9]) - 1];									
									info.TopZ = vw.TopZ;
									info.BottomZ = vw.BottomZ;
									info.Intercept.Z = vw.BottomZ;
									bottomlists[vw.Id].Add(bottomtks[id - 1] = new SySal.OperaDb.Scanning.MIPIndexedEmulsionTrack(info, id - 1, id, vw));
								}
								else
								{
									View vw = topviews[Convert.ToInt32(dr[9]) - 1];
									info.TopZ = vw.TopZ;
									info.BottomZ = vw.BottomZ;
									info.Intercept.Z = vw.TopZ;
									toplists[vw.Id].Add(toptks[id - 1] = new SySal.OperaDb.Scanning.MIPIndexedEmulsionTrack(info, id - 1, id, vw));
								}
							}

							for (i = 0; i < toplists.Length; i++)
							{
								topviews[i].SetTracks((SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack [])(toplists[i].ToArray(typeof(SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack))));
								toplists[i] = null;
							}
							for (i = 0; i < bottomlists.Length; i++)
							{
								bottomviews[i].SetTracks((SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack [])(bottomlists[i].ToArray(typeof(SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack))));
								bottomlists[i] = null;
							}
							toplists = null;
							bottomlists = null;

							ds = new DataSet();
							da = new OracleDataAdapter("SELECT /*+INDEX(TB_MIPBASETRACKS PK_MIPBASETRACKS) */ ID, POSX, POSY, SLOPEX, SLOPEY, NVL(GRAINS, FLOOR(PH * 0.001)), NVL(AREASUM, MOD(PH, 1000)), SIGMA, ID_DOWNTRACK, ID_UPTRACK FROM TB_MIPBASETRACKS WHERE(TB_MIPBASETRACKS.ID_EVENTBRICK = " + db_brick_id + " AND TB_MIPBASETRACKS.ID_ZONE = " + db_id + ")", oracleconn);
							da.Fill(ds);
							dt = ds.Tables[0];
							m_Tracks = new SySal.OperaDb.Scanning.MIPBaseTrack[dt.Rows.Count];
							foreach (DataRow dr in dt.Rows)
							{
								SySal.Tracking.MIPEmulsionTrackInfo info = new SySal.Tracking.MIPEmulsionTrackInfo();
								info.Intercept.X = Convert.ToDouble(dr[1]);
								info.Intercept.Y = Convert.ToDouble(dr[2]);
								SySal.OperaDb.Scanning.MIPIndexedEmulsionTrack toptk = toptks[Convert.ToInt64(dr[8]) - 1];
								SySal.OperaDb.Scanning.MIPIndexedEmulsionTrack bottk = bottomtks[Convert.ToInt64(dr[9]) - 1];
								info.Intercept.Z = toptk.View.BottomZ;
								info.Slope.X = Convert.ToDouble(dr[3]);
								info.Slope.Y = Convert.ToDouble(dr[4]);
								info.Slope.Z = 1.0;
								info.Count = Convert.ToUInt16(dr[5]);
								info.AreaSum = Convert.ToUInt32(dr[6]);
								info.Sigma = Convert.ToDouble(dr[7]);
								info.TopZ = toptk.View.TopZ;
								info.BottomZ = bottk.View.BottomZ;
								int id = Convert.ToInt32(dr[0]);
								m_Tracks[id - 1] = new SySal.OperaDb.Scanning.MIPBaseTrack(info, (int)id - 1, (int)id, toptk, bottk);
							}
							topside.SetTracks(toptks);
							bottomside.SetTracks(bottomtks);
						}						
#endregion
					}
						break;

					default: throw new Exception("Unsupported schema version " + conn.DBSchemaVersion);
				}
			}

			const int BatchSize = 1000;
			static long [] a_idbrick = new long[BatchSize];
			static long [] a_idlz = new long[BatchSize];
			static long [] a_newid = new long[BatchSize];
			static double [] a_posx = new double[BatchSize];
			static double [] a_posy = new double[BatchSize];
			static double [] a_slopex = new double[BatchSize];
			static double [] a_slopey = new double[BatchSize];
			static double [] a_sigma = new double[BatchSize];
			static int [] a_areasum = new int[BatchSize];
			static int [] a_grains = new int[BatchSize];
			static int [] a_side = new int[BatchSize];
			static int [] a_idview = new int[BatchSize];
			static double [] a_downz = new double[BatchSize];
			static double [] a_upz = new double[BatchSize];
			static long [] a_iddown = new long[BatchSize];
			static long [] a_idup = new long[BatchSize];
			static OracleCommand cmdb = null;
			static OracleCommand cmdm1 = null;
			static OracleCommand cmdm2 = null;
			static OracleCommand cmdv = null;

			/// <summary>
			/// Saves a LinkedZone to the specified OperaDb and associates it to a specified batch.
			/// </summary>
			/// <param name="lz">the LinkedZone to be saved.</param>
			/// <param name="db_brick_id">the brick to which the LinkedZone belongs.</param>
			/// <param name="db_plate_id">the plate to which the LinkedZone belongs.</param>
			/// <param name="db_procop_id">the process operation that required this LinkedZone to be scanned.</param>
			/// <param name="series">the series number of the zone.</param>
			/// <param name="rawdatapath">the path to the raw data files from which the linked zone originates.</param>
			/// <param name="starttime">date and time when the scanning started for this linked zone.</param>
			/// <param name="endtime">date and time when the scanning ended for this linked zone.</param>
			/// <param name="conn">the DB connection to be used.</param>
			/// <param name="trans">the DB transaction to be used. Should not be null, since the TotalScan reconstruction usually needs several tables.</param>
			/// <returns>the DB identifier that has been assigned to the stored LinkedZone.</returns>
			public static long Save(SySal.Scanning.Plate.IO.OPERA.LinkedZone lz, long db_brick_id, long db_plate_id, long db_procop_id, long series, string rawdatapath, DateTime starttime, DateTime endtime, OperaDbConnection conn, OperaDbTransaction trans)
			{
				lock(a_idbrick)
				{
					switch (conn.DBSchemaVersion)
					{
						case DBSchemaVersion.Basic_V1:
						{
#region basic_v1_write
							int s, i, j, k, b, n;
							OracleConnection oracleconn = (OracleConnection)conn.Conn;
							long idlz;
							double basez;
							SySal.Scanning.Plate.Side side;

							OracleCommand cmdz = new OracleCommand("INSERT INTO TB_ZONES (ID_EVENTBRICK, ID_PLATE, ID_PROCESSOPERATION, SERIES, MINX, MAXX, MINY, MAXY, RAWDATAPATH, STARTTIME, ENDTIME, BASETHICKNESS) VALUES (:brickid, :plateid, :procopid, :series, :minx, :maxx, :miny, :maxy, :rawdatapath, :starttime, :endtime, :basethickness) RETURNING TB_ZONES.ID INTO :newid", oracleconn);
							cmdz.CommandType = CommandType.Text;
							cmdz.Parameters.Add("brickid", OracleDbType.Int64, System.Data.ParameterDirection.Input).Value = db_brick_id;
							cmdz.Parameters.Add("plateid", OracleDbType.Int64, System.Data.ParameterDirection.Input).Value = db_plate_id;
							cmdz.Parameters.Add("procopid", OracleDbType.Int64, System.Data.ParameterDirection.Input).Value = db_procop_id;
							cmdz.Parameters.Add("series", OracleDbType.Int64, System.Data.ParameterDirection.Input).Value = series;
							cmdz.Parameters.Add("minx", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = lz.Extents.MinX;
							cmdz.Parameters.Add("maxx", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = lz.Extents.MaxX;
							cmdz.Parameters.Add("miny", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = lz.Extents.MinY;
							cmdz.Parameters.Add("maxy", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = lz.Extents.MaxY;
							cmdz.Parameters.Add("rawdatapath", OracleDbType.Varchar2, System.Data.ParameterDirection.Input).Value = rawdatapath;
							cmdz.Parameters.Add("starttime", OracleDbType.TimeStamp, System.Data.ParameterDirection.Input).Value = starttime;
							cmdz.Parameters.Add("endtime", OracleDbType.TimeStamp, System.Data.ParameterDirection.Input).Value = endtime;
							cmdz.Parameters.Add("basethickness", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = lz.Top.BottomZ - lz.Bottom.TopZ;
							cmdz.Parameters.Add("newid", OracleDbType.Int64, System.Data.ParameterDirection.Output);
							cmdz.ExecuteNonQuery();
							idlz = Convert.ToInt64(cmdz.Parameters[12].Value);
							for (i = 0; i < BatchSize; i++)
							{
								a_idbrick[i] = db_brick_id;
								a_idlz[i] = idlz;
							}

							if (cmdm1 == null)
							{
								cmdm1 = new OracleCommand("INSERT INTO TB_MIPMICROTRACKS (ID_EVENTBRICK, ID_ZONE, ID, POSX, POSY, SLOPEX, SLOPEY, SIGMA, AREASUM, GRAINS, SIDE) VALUES (:brickid, :idlz, :newid, :posx, :posy, :slopex, :slopey, :sigma, :areasum, :grains, :side)", oracleconn);
								cmdm1.CommandType = CommandType.Text;
								cmdm1.Parameters.Add("idbrick", OracleDbType.Int64, System.Data.ParameterDirection.Input).Value = a_idbrick;
								cmdm1.Parameters.Add("idlz", OracleDbType.Int64, System.Data.ParameterDirection.Input).Value = a_idlz;
								cmdm1.Parameters.Add("newid", OracleDbType.Int64, System.Data.ParameterDirection.Input).Value = a_newid;
								cmdm1.Parameters.Add("posx", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = a_posx;
								cmdm1.Parameters.Add("posy", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = a_posy;
								cmdm1.Parameters.Add("slopex", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = a_slopex;
								cmdm1.Parameters.Add("slopey", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = a_slopey;
								cmdm1.Parameters.Add("sigma", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = a_sigma;
								cmdm1.Parameters.Add("areasum", OracleDbType.Int32, System.Data.ParameterDirection.Input).Value = a_areasum;
								cmdm1.Parameters.Add("grains", OracleDbType.Int32, System.Data.ParameterDirection.Input).Value = a_grains;
								cmdm1.Parameters.Add("side", OracleDbType.Int32, System.Data.ParameterDirection.Input).Value = a_side;
							}
							else cmdm1.Connection = oracleconn;
							cmdm1.Prepare();

				
							for (s = 0; s < 2; s++)
							{
								if (s == 0)
								{
									side = lz.Top;
									basez = lz.Top.BottomZ;
								}
								else
								{
									side = lz.Bottom;
									basez = lz.Bottom.TopZ;
								}
								n = side.Length;
								for (i = 0; i < n; i += BatchSize)
								{
									b = i + BatchSize; if (b > n) b = n;
									for (j = i; j < b; j++)
									{
										k = j - i;
										SySal.Tracking.MIPEmulsionTrackInfo info = side[j].Info;
										double dz = (basez - info.Intercept.Z);
										a_newid[k] = j + 1;
										a_posx[k] = info.Intercept.X + info.Slope.X * dz;
										a_posy[k] = info.Intercept.Y + info.Slope.Y * dz;
										a_slopex[k] = info.Slope.X;
										a_slopey[k] = info.Slope.Y;
										a_sigma[k] = info.Sigma;
										a_areasum[k] = (int)info.AreaSum;
										a_grains[k] = info.Count;
										a_side[k] = s + 1;
									}
									cmdm1.ArrayBindCount = (b - i);
									cmdm1.ExecuteNonQuery();
								}
							}
							cmdm1.Connection = null;
							//cmdm1.Dispose();

							n = lz.Length;
							if (cmdb == null)
							{
								cmdb = new OracleCommand("INSERT INTO TB_MIPBASETRACKS (ID_EVENTBRICK, ID_ZONE, ID, POSX, POSY, SLOPEX, SLOPEY, SIGMA, AREASUM, GRAINS, ID_DOWNTRACK, ID_UPTRACK, ID_DOWNSIDE, ID_UPSIDE) VALUES (:brickid, :idlz, :newid, :posx, :posy, :slopex, :slopey, :sigma, :areasum, :grains, :iddown, :idup, 1, 2)", oracleconn);
								cmdb.CommandType = CommandType.Text;
								cmdb.Parameters.Add("idbrick", OracleDbType.Int64, System.Data.ParameterDirection.Input).Value = a_idbrick;
								cmdb.Parameters.Add("idlz", OracleDbType.Int64, System.Data.ParameterDirection.Input).Value = a_idlz;
								cmdb.Parameters.Add("newid", OracleDbType.Int64, System.Data.ParameterDirection.Input).Value = a_newid;
								cmdb.Parameters.Add("posx", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = a_posx;
								cmdb.Parameters.Add("posy", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = a_posy;
								cmdb.Parameters.Add("slopex", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = a_slopex;
								cmdb.Parameters.Add("slopey", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = a_slopey;
								cmdb.Parameters.Add("sigma", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = a_sigma;
								cmdb.Parameters.Add("areasum", OracleDbType.Int32, System.Data.ParameterDirection.Input).Value = a_areasum;
								cmdb.Parameters.Add("grains", OracleDbType.Int32, System.Data.ParameterDirection.Input).Value = a_grains;
								cmdb.Parameters.Add("iddown", OracleDbType.Int64, System.Data.ParameterDirection.Input).Value = a_iddown;
								cmdb.Parameters.Add("idup", OracleDbType.Int64, System.Data.ParameterDirection.Input).Value = a_idup;
							}
							else cmdb.Connection = oracleconn;
							cmdb.Prepare();

							basez = lz.Top.BottomZ;
							for (i = 0; i < n; i += BatchSize)
							{
								b = i + BatchSize; if (b > n) b = n;
								for (j = i; j < b; j++)
								{
									k = j - i;
									SySal.Tracking.MIPEmulsionTrackInfo info = lz[j].Info;
									double dz = (basez - info.Intercept.Z);
									a_newid[k] = j + 1;
									a_posx[k] = info.Intercept.X + info.Slope.X * dz;
									a_posy[k] = info.Intercept.Y + info.Slope.Y * dz;
									a_slopex[k] = info.Slope.X;
									a_slopey[k] = info.Slope.Y;
									a_sigma[k] = info.Sigma;
									a_areasum[k] = (int)info.AreaSum;
									a_grains[k] = info.Count;
									a_iddown[k] = lz[j].Top.Id + 1;
									a_idup[k] = lz[j].Bottom.Id + 1;
								}
								cmdb.ArrayBindCount = (b - i);
								cmdb.ExecuteNonQuery();
							}
							cmdb.Connection = null;
							//cmdb.Dispose();
							return idlz;
#endregion
						}
							break;

						case DBSchemaVersion.HasViews_V2:
						{
#region hasviews_v2_write
							int s, i, j, k, b, n;
							OracleConnection oracleconn = (OracleConnection)conn.Conn;
							long idlz;
							double basez;
							SySal.Scanning.Plate.Side side;

                            OracleCommand cmdz = new OracleCommand(
                                "INSERT INTO " + (conn.HasBufferTables ? "OPERA.LZ_ZONES" : "TB_ZONES") + " (ID_EVENTBRICK, ID_PLATE, ID_PROCESSOPERATION, SERIES, MINX, MAXX, MINY, MAXY, RAWDATAPATH, STARTTIME, ENDTIME, TXX, TXY, TYX, TYY, TDX, TDY) VALUES (:brickid, :plateid, :procopid, :series, :minx, :maxx, :miny, :maxy, :rawdatapath, :starttime, :endtime, :txx, :txy, :tyx, :tyy, :tdx, :tdy) RETURNING " + (conn.HasBufferTables ? "OPERA.LZ_ZONES" : "TB_ZONES") + ".ID INTO :newid", 
                                oracleconn);
							cmdz.CommandType = CommandType.Text;
							cmdz.Parameters.Add("brickid", OracleDbType.Int64, System.Data.ParameterDirection.Input).Value = db_brick_id;
							cmdz.Parameters.Add("plateid", OracleDbType.Int64, System.Data.ParameterDirection.Input).Value = db_plate_id;
							cmdz.Parameters.Add("procopid", OracleDbType.Int64, System.Data.ParameterDirection.Input).Value = db_procop_id;
							cmdz.Parameters.Add("series", OracleDbType.Int64, System.Data.ParameterDirection.Input).Value = series;
							cmdz.Parameters.Add("minx", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = lz.Extents.MinX;
							cmdz.Parameters.Add("maxx", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = lz.Extents.MaxX;
							cmdz.Parameters.Add("miny", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = lz.Extents.MinY;
							cmdz.Parameters.Add("maxy", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = lz.Extents.MaxY;
							cmdz.Parameters.Add("rawdatapath", OracleDbType.Varchar2, System.Data.ParameterDirection.Input).Value = rawdatapath;
							cmdz.Parameters.Add("starttime", OracleDbType.TimeStamp, System.Data.ParameterDirection.Input).Value = starttime;
							cmdz.Parameters.Add("endtime", OracleDbType.TimeStamp, System.Data.ParameterDirection.Input).Value = endtime;
							SySal.DAQSystem.Scanning.IntercalibrationInfo transform = lz.Transform;
							cmdz.Parameters.Add("txx", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = transform.MXX;
							cmdz.Parameters.Add("txy", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = transform.MXY;
							cmdz.Parameters.Add("tyx", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = transform.MYX;
							cmdz.Parameters.Add("tyy", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = transform.MYY;
							cmdz.Parameters.Add("tdx", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = transform.TX - transform.MXX * transform.RX - transform.MXY * transform.RY;
							cmdz.Parameters.Add("tdy", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = transform.TY - transform.MYX * transform.RX - transform.MYY * transform.RY;
							cmdz.Parameters.Add("newid", OracleDbType.Int64, System.Data.ParameterDirection.Output);
							cmdz.ExecuteNonQuery();
							idlz = Convert.ToInt64(cmdz.Parameters[17].Value);
							for (i = 0; i < BatchSize; i++)
							{
								a_idbrick[i] = db_brick_id;
								a_idlz[i] = idlz;
							}

							if (cmdv == null)
							{
                                cmdv = new OracleCommand("INSERT INTO " + (conn.HasBufferTables ? "OPERA.LZ_VIEWS" : "TB_VIEWS") + " (ID_EVENTBRICK, ID_ZONE, SIDE, ID, DOWNZ, UPZ, POSX, POSY) VALUES (:brickid, :idlz, :side, :newid, :downz, :upz, :posx, :posy)", oracleconn);
								cmdv.CommandType = CommandType.Text;
								cmdv.Parameters.Add("idbrick", OracleDbType.Int64, System.Data.ParameterDirection.Input).Value = a_idbrick;
								cmdv.Parameters.Add("idlz", OracleDbType.Int64, System.Data.ParameterDirection.Input).Value = a_idlz;
								cmdv.Parameters.Add("side", OracleDbType.Int64, System.Data.ParameterDirection.Input).Value = a_side;
								cmdv.Parameters.Add("newid", OracleDbType.Int64, System.Data.ParameterDirection.Input).Value = a_newid;
								cmdv.Parameters.Add("downz", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = a_downz;
								cmdv.Parameters.Add("upz", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = a_upz;
								cmdv.Parameters.Add("posx", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = a_posx;
								cmdv.Parameters.Add("posy", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = a_posy;
							}
							else cmdv.Connection = oracleconn;
							cmdv.Prepare();

							for (s = 0; s < 2; s++)
							{
								if (s == 0)
								{
									side = lz.Top;
									basez = lz.Top.BottomZ;
								}
								else
								{
									side = lz.Bottom;
									basez = lz.Bottom.TopZ;
								}
								n = ((SySal.Scanning.Plate.IO.OPERA.LinkedZone.Side)side).ViewCount;
								for (i = 0; i < n; i += BatchSize)
								{
									b = i + BatchSize; if (b > n) b = n;
									for (j = i; j < b; j++)
									{
										k = j - i;
										SySal.Scanning.Plate.IO.OPERA.LinkedZone.View vw = ((SySal.Scanning.Plate.IO.OPERA.LinkedZone.Side)side).View(j);
										a_side[k] = s + 1;										
										a_newid[k] = j + 1;
										a_downz[k] = vw.TopZ;
										a_upz[k] = vw.BottomZ;
										a_posx[k] = vw.Position.X;
										a_posy[k] = vw.Position.Y;
									}
									cmdv.ArrayBindCount = (b - i);
									cmdv.ExecuteNonQuery();
								}
							}
							cmdv.Connection = null;
							//cmdv.Dispose();
														
							if (cmdm2 == null)
							{
                                cmdm2 = new OracleCommand("INSERT INTO " + (conn.HasBufferTables ? "OPERA.LZ_MIPMICROTRACKS" : "TB_MIPMICROTRACKS") + " (ID_EVENTBRICK, ID_ZONE, ID, POSX, POSY, SLOPEX, SLOPEY, SIGMA, AREASUM, GRAINS, SIDE, ID_VIEW) VALUES (:brickid, :idlz, :newid, :posx, :posy, :slopex, :slopey, :sigma, :areasum, :grains, :side, :idview)", oracleconn);
								cmdm2.CommandType = CommandType.Text;
								cmdm2.Parameters.Add("idbrick", OracleDbType.Int64, System.Data.ParameterDirection.Input).Value = a_idbrick;
								cmdm2.Parameters.Add("idlz", OracleDbType.Int64, System.Data.ParameterDirection.Input).Value = a_idlz;
								cmdm2.Parameters.Add("newid", OracleDbType.Int64, System.Data.ParameterDirection.Input).Value = a_newid;
								cmdm2.Parameters.Add("posx", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = a_posx;
								cmdm2.Parameters.Add("posy", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = a_posy;
								cmdm2.Parameters.Add("slopex", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = a_slopex;
								cmdm2.Parameters.Add("slopey", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = a_slopey;
								cmdm2.Parameters.Add("sigma", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = a_sigma;
								cmdm2.Parameters.Add("areasum", OracleDbType.Int32, System.Data.ParameterDirection.Input).Value = a_areasum;
								cmdm2.Parameters.Add("grains", OracleDbType.Int32, System.Data.ParameterDirection.Input).Value = a_grains;
								cmdm2.Parameters.Add("side", OracleDbType.Int32, System.Data.ParameterDirection.Input).Value = a_side;
								cmdm2.Parameters.Add("idview", OracleDbType.Int32, System.Data.ParameterDirection.Input).Value = a_idview;
							}
							else cmdm2.Connection = oracleconn;
							cmdm2.Prepare();
				
							for (s = 0; s < 2; s++)
							{
								if (s == 0)
								{
									side = lz.Top;
									basez = lz.Top.BottomZ;
								}
								else
								{
									side = lz.Bottom;
									basez = lz.Bottom.TopZ;
								}
								n = side.Length;
								for (i = 0; i < n; i += BatchSize)
								{
									b = i + BatchSize; if (b > n) b = n;
									for (j = i; j < b; j++)
									{
										k = j - i;
										SySal.Tracking.MIPEmulsionTrackInfo info = side[j].Info;
										double dz = (basez - info.Intercept.Z);
										a_newid[k] = j + 1;
										a_posx[k] = info.Intercept.X + info.Slope.X * dz;
										a_posy[k] = info.Intercept.Y + info.Slope.Y * dz;
										a_slopex[k] = info.Slope.X;
										a_slopey[k] = info.Slope.Y;
										a_sigma[k] = info.Sigma;
										a_areasum[k] = (int)info.AreaSum;
										a_grains[k] = info.Count;
										a_side[k] = s + 1;
										a_idview[k] = ((SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack)(side[j])).View.Id + 1;
									}
									cmdm2.ArrayBindCount = (b - i);
									cmdm2.ExecuteNonQuery();
								}
							}
							cmdm2.Connection = null;
							//cmdm2.Dispose();

							n = lz.Length;
							if (cmdb == null)
							{
                                cmdb = new OracleCommand("INSERT INTO " + (conn.HasBufferTables ? "OPERA.LZ_MIPBASETRACKS" : "TB_MIPBASETRACKS") + " (ID_EVENTBRICK, ID_ZONE, ID, POSX, POSY, SLOPEX, SLOPEY, SIGMA, AREASUM, GRAINS, ID_DOWNTRACK, ID_UPTRACK, ID_DOWNSIDE, ID_UPSIDE) VALUES (:brickid, :idlz, :newid, :posx, :posy, :slopex, :slopey, :sigma, :areasum, :grains, :iddown, :idup, 1, 2)", oracleconn);
								cmdb.CommandType = CommandType.Text;
								cmdb.Parameters.Add("idbrick", OracleDbType.Int64, System.Data.ParameterDirection.Input).Value = a_idbrick;
								cmdb.Parameters.Add("idlz", OracleDbType.Int64, System.Data.ParameterDirection.Input).Value = a_idlz;
								cmdb.Parameters.Add("newid", OracleDbType.Int64, System.Data.ParameterDirection.Input).Value = a_newid;
								cmdb.Parameters.Add("posx", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = a_posx;
								cmdb.Parameters.Add("posy", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = a_posy;
								cmdb.Parameters.Add("slopex", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = a_slopex;
								cmdb.Parameters.Add("slopey", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = a_slopey;
								cmdb.Parameters.Add("sigma", OracleDbType.Double, System.Data.ParameterDirection.Input).Value = a_sigma;
								cmdb.Parameters.Add("areasum", OracleDbType.Int32, System.Data.ParameterDirection.Input).Value = a_areasum;
								cmdb.Parameters.Add("grains", OracleDbType.Int32, System.Data.ParameterDirection.Input).Value = a_grains;
								cmdb.Parameters.Add("iddown", OracleDbType.Int64, System.Data.ParameterDirection.Input).Value = a_iddown;
								cmdb.Parameters.Add("idup", OracleDbType.Int64, System.Data.ParameterDirection.Input).Value = a_idup;
							}
							else cmdb.Connection = oracleconn;
							cmdb.Prepare();

							for (i = 0; i < n; i += BatchSize)
							{
								b = i + BatchSize; if (b > n) b = n;
								for (j = i; j < b; j++)
								{
									k = j - i;
									SySal.Scanning.MIPBaseTrack bt = lz[j];
									SySal.Tracking.MIPEmulsionTrackInfo info = bt.Info;
									basez = ((SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack)(bt.Top)).View.BottomZ;
									double dz = (basez - info.Intercept.Z);
									a_newid[k] = j + 1;
									a_posx[k] = info.Intercept.X + info.Slope.X * dz;
									a_posy[k] = info.Intercept.Y + info.Slope.Y * dz;
									a_slopex[k] = info.Slope.X;
									a_slopey[k] = info.Slope.Y;
									a_sigma[k] = info.Sigma;
									a_areasum[k] = (int)info.AreaSum;
									a_grains[k] = info.Count;
									a_iddown[k] = lz[j].Top.Id + 1;
									a_idup[k] = lz[j].Bottom.Id + 1;
								}
								cmdb.ArrayBindCount = (b - i);
								cmdb.ExecuteNonQuery();
							}
							cmdb.Connection = null;
							//cmdb.Dispose();
							return idlz;
#endregion
						}
							break;

						default: throw new Exception("Unsupported schema version " + conn.DBSchemaVersion);
					}

				}				
			}
		}

        /// <summary>
        /// A class for serializing/deserializing <see cref="SySal.OperaDb.TotalScan.DBMIPMicroTrackIndex"/> objects to TLG files.
        /// </summary>
        public class DBMIPMicroTrackIndex
        {
            /// <summary>
            /// Section tag for this class in multi-section TLG files.
            /// </summary>
            public const byte SectionTag = 0x03;
            /// <summary>
            /// Index of DB Ids for microtracks on top side.
            /// </summary>
            public SySal.OperaDb.TotalScan.DBMIPMicroTrackIndex[] TopTracksIndex;
            /// <summary>
            /// Index of DB Ids for microtracks on bottom side.
            /// </summary>
            public SySal.OperaDb.TotalScan.DBMIPMicroTrackIndex[] BottomTracksIndex;
            /// <summary>
            /// Builds an empty DBMIPMicroTrackIndex
            /// </summary>
            public DBMIPMicroTrackIndex() 
            {
                TopTracksIndex = new SySal.OperaDb.TotalScan.DBMIPMicroTrackIndex[0];
                BottomTracksIndex = new SySal.OperaDb.TotalScan.DBMIPMicroTrackIndex[0];
            }
            /// <summary>
            /// Saves a pair of index arrays to a section in a TLG file.
            /// </summary>
            /// <param name="str">the stream to which data are to be written.</param>
            public void Save(System.IO.Stream str)
            {
                if (SySal.Scanning.Plate.IO.OPERA.LinkedZone.FindSection(str, SectionTag, false)) throw new Exception("DBMIPMicroTrackIndex Section already exists in the stream!");
                System.IO.BinaryWriter w = new System.IO.BinaryWriter(str);
                long nextsectionpos = 0;
                w.Write(SectionTag);
                long nextsectionref = str.Position;
                w.Write(nextsectionpos);
                w.Write(TopTracksIndex.Length);
                w.Write(BottomTracksIndex.Length);
                foreach (SySal.OperaDb.TotalScan.DBMIPMicroTrackIndex it in TopTracksIndex)
                    it.Write(w);
                foreach (SySal.OperaDb.TotalScan.DBMIPMicroTrackIndex ib in BottomTracksIndex)
                    ib.Write(w);
                w.Flush();
                nextsectionpos = str.Position;
                str.Position = nextsectionref;
                w.Write(nextsectionpos);
                w.Flush();
                str.Position = nextsectionpos;
            }
            /// <summary>
            /// Builds a DBMIPMicroTrackIndex from the proper section in a TLG file. An exception is thrown if the section is not found or is corrupt.
            /// </summary>
            /// <param name="str">the stream from which data are to be read.</param>
            public DBMIPMicroTrackIndex(System.IO.Stream str)
            {
                if (!SySal.Scanning.Plate.IO.OPERA.LinkedZone.FindSection(str, SectionTag, true)) throw new Exception("No DBMIPMicroTrackIndex section found in stream!");
                System.IO.BinaryReader r = new System.IO.BinaryReader(str);
                TopTracksIndex = new SySal.OperaDb.TotalScan.DBMIPMicroTrackIndex[r.ReadInt32()];
                BottomTracksIndex = new SySal.OperaDb.TotalScan.DBMIPMicroTrackIndex[r.ReadInt32()];
                int i;
                for (i = 0; i < TopTracksIndex.Length; i++)
                    TopTracksIndex[i] = (SySal.OperaDb.TotalScan.DBMIPMicroTrackIndex)SySal.OperaDb.TotalScan.DBMIPMicroTrackIndex.CreateFromReader(r);
                for (i = 0; i < BottomTracksIndex.Length; i++)
                    BottomTracksIndex[i] = (SySal.OperaDb.TotalScan.DBMIPMicroTrackIndex)SySal.OperaDb.TotalScan.DBMIPMicroTrackIndex.CreateFromReader(r);
            }
        }

        /// <summary>
        /// A class that contains general purpose procedures.
        /// </summary>
        public class Utilities
        {
            public static char MarkTypeToChar(SySal.DAQSystem.Drivers.MarkType mt)
            {
                switch (mt)
                {
                    case SySal.DAQSystem.Drivers.MarkType.SpotOptical: return SySal.DAQSystem.Drivers.MarkChar.SpotOptical;
                    case SySal.DAQSystem.Drivers.MarkType.SpotXRay: return SySal.DAQSystem.Drivers.MarkChar.SpotXRay;
                    case SySal.DAQSystem.Drivers.MarkType.LineXRay: return SySal.DAQSystem.Drivers.MarkChar.LineXRay;
                    case SySal.DAQSystem.Drivers.MarkType.None: return SySal.DAQSystem.Drivers.MarkChar.None;
                    default: throw new Exception("Unsupported mark type.");
                }
            }

            public static SySal.DAQSystem.Drivers.MarkType CharToMarkType(char mt)
            {
                switch (mt)
                {
                    case SySal.DAQSystem.Drivers.MarkChar.SpotOptical: return SySal.DAQSystem.Drivers.MarkType.SpotOptical;
                    case SySal.DAQSystem.Drivers.MarkChar.SpotXRay: return SySal.DAQSystem.Drivers.MarkType.SpotXRay;
                    case SySal.DAQSystem.Drivers.MarkChar.LineXRay: return SySal.DAQSystem.Drivers.MarkType.LineXRay;
                    case SySal.DAQSystem.Drivers.MarkChar.None: return SySal.DAQSystem.Drivers.MarkType.None;
                    default: throw new Exception("Unsupported mark type.");
                }
            }

            public static string MarkTypeToString(SySal.DAQSystem.Drivers.MarkType mt)
            {
                string str = "";
                if ((mt & SySal.DAQSystem.Drivers.MarkType.SpotOptical) != SySal.DAQSystem.Drivers.MarkType.None)
                {
                    str += SySal.DAQSystem.Drivers.MarkChar.SpotOptical;
                    mt -= SySal.DAQSystem.Drivers.MarkType.SpotOptical;
                }
                if ((mt & SySal.DAQSystem.Drivers.MarkType.SpotXRay) != SySal.DAQSystem.Drivers.MarkType.None)
                {
                    str += SySal.DAQSystem.Drivers.MarkChar.SpotXRay;
                    mt -= SySal.DAQSystem.Drivers.MarkType.SpotXRay;
                }
                if ((mt & SySal.DAQSystem.Drivers.MarkType.LineXRay) != SySal.DAQSystem.Drivers.MarkType.None)
                {
                    str += SySal.DAQSystem.Drivers.MarkChar.LineXRay;
                    mt -= SySal.DAQSystem.Drivers.MarkType.LineXRay;
                }
                if (mt != SySal.DAQSystem.Drivers.MarkType.None)
                    throw new Exception("Unsupported mark type.");
                return str;
            }

			struct Mark
			{
				public long IdTemplateMark;
				public int MarkId;
				public double TemplateX, TemplateY;
				public double X, Y;
				public int Side;
                public SySal.DAQSystem.Drivers.MarkType ShapeType;
			}
			
			/// <summary>
			/// Retrieves the map string for a plate.
			/// </summary>
			/// <param name="brickid">the brick to which the plate belongs.</param>
			/// <param name="plateid">the plate for which the map string is sought.</param>
			/// <param name="calibrated">true if the calibrated map is sought, false if nominal marks are to be used.</param>
            /// <param name="shapetype">shape and type of the mark set.</param>
			/// <param name="calibrationid">the Id of the calibration used or zero if no calibration is found/sought.</param>
			/// <param name="conn">the DB connection to be used.</param>
			/// <param name="trans">the DB transaction in which the command is to be executed.</param>
			/// <returns>the map string for the plate.</returns>
            public static string GetMapString(long brickid, long plateid, bool calibrated, SySal.DAQSystem.Drivers.MarkType shapetype, out long calibrationid, OperaDbConnection conn, OperaDbTransaction trans)
			{

				System.Globalization.CultureInfo InvC = System.Globalization.CultureInfo.InvariantCulture;
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT MINX - ZEROX, MAXX - ZEROX, MINY - ZEROY, MAXY - ZEROY FROM TB_EVENTBRICKS WHERE ID = " + brickid, conn, trans).Fill(ds);
				SySal.DAQSystem.Scanning.IntercalibrationInfo intercal = new SySal.DAQSystem.Scanning.IntercalibrationInfo();
				double MinX = Convert.ToDouble(ds.Tables[0].Rows[0][0]);
				double MaxX = Convert.ToDouble(ds.Tables[0].Rows[0][1]);
				double MinY = Convert.ToDouble(ds.Tables[0].Rows[0][2]);
				double MaxY = Convert.ToDouble(ds.Tables[0].Rows[0][3]);
				intercal.RX = (MinX + MaxX) * 0.5;
				intercal.RY = (MinY + MaxY) * 0.5;

				if (calibrated)
				{
					ds = new System.Data.DataSet();
					new SySal.OperaDb.OperaDbDataAdapter("SELECT MAPXX, MAPXY, MAPYX, MAPYY, MAPDX, MAPDY, NVL(CALIBRATION, 0) FROM VW_PLATES WHERE ID_EVENTBRICK = " + brickid + " AND ID = " + plateid, conn, trans).Fill(ds);
					intercal.MXX = Convert.ToDouble(ds.Tables[0].Rows[0][0]);
					intercal.MXY = Convert.ToDouble(ds.Tables[0].Rows[0][1]);
					intercal.MYX = Convert.ToDouble(ds.Tables[0].Rows[0][2]);
					intercal.MYY = Convert.ToDouble(ds.Tables[0].Rows[0][3]);
					intercal.TX = Convert.ToDouble(ds.Tables[0].Rows[0][4]);
					intercal.TY = Convert.ToDouble(ds.Tables[0].Rows[0][5]);
					intercal.TZ = 0.0;
					calibrationid = Convert.ToInt64(ds.Tables[0].Rows[0][6]);
                    if (calibrationid > 0)
                        shapetype = SySal.OperaDb.Scanning.Utilities.CharToMarkType(System.Convert.ToChar(new SySal.OperaDb.OperaDbCommand("SELECT MARKSETS FROM TB_PLATE_CALIBRATIONS WHERE ID_EVENTBRICK = " + brickid + " AND ID_PLATE = " + plateid + " AND ID_PROCESSOPERATION = " + calibrationid, conn, trans).ExecuteScalar().ToString()));
				}
				else
				{
					intercal.MXX = intercal.MYY = 1.0;
					intercal.MXY = intercal.MYX = 0.0;
					intercal.TX = intercal.TY = intercal.TZ = 0.0;
					calibrationid = 0;
				}

				ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID, ID_MARK, POSX, POSY, SIDE FROM TB_TEMPLATEMARKSETS WHERE (ID_EVENTBRICK = " + brickid + ") AND SHAPE = '" + MarkTypeToChar(shapetype) + "' ORDER BY ID_MARK ASC", conn, trans).Fill(ds);
				Mark [] MarkSet = new Mark[ds.Tables[0].Rows.Count];
                string mapstring = ((shapetype == SySal.DAQSystem.Drivers.MarkType.LineXRay) ? "mapX: " : "mapext: ") + brickid + " " + plateid + " 0 0; " + MarkSet.Length + " " + MinX + " " + MinY + " " + MaxX + " " + MaxY;
				int i;
				for (i = 0; i < ds.Tables[0].Rows.Count; i++)
				{
					MarkSet[i].IdTemplateMark = SySal.OperaDb.Convert.ToInt64(ds.Tables[0].Rows[i][0]);
					MarkSet[i].MarkId = SySal.OperaDb.Convert.ToInt32(ds.Tables[0].Rows[i][1]);
					MarkSet[i].TemplateX = SySal.OperaDb.Convert.ToDouble(ds.Tables[0].Rows[i][2]);
					MarkSet[i].TemplateY = SySal.OperaDb.Convert.ToDouble(ds.Tables[0].Rows[i][3]);
					MarkSet[i].X = intercal.MXX * (MarkSet[i].TemplateX - intercal.RX) + intercal.MXY * (MarkSet[i].TemplateY - intercal.RY) + intercal.TX + intercal.RX;
					MarkSet[i].Y = intercal.MYX * (MarkSet[i].TemplateX - intercal.RX) + intercal.MYY * (MarkSet[i].TemplateY - intercal.RY) + intercal.TY + intercal.RY;
					MarkSet[i].Side = SySal.OperaDb.Convert.ToInt32(ds.Tables[0].Rows[i][4]);
					if (calibrated) mapstring += "; " + MarkSet[i].MarkId + " " + MarkSet[i].X + " " + MarkSet[i].Y + " " + MarkSet[i].TemplateX + " " + MarkSet[i].TemplateY + " 1 1 " + MarkSet[i].Side; 
					else mapstring += "; " + MarkSet[i].MarkId + " " + MarkSet[i].TemplateX + " " + MarkSet[i].TemplateY + " " + MarkSet[i].TemplateX + " " + MarkSet[i].TemplateY + " 1 1 " + MarkSet[i].Side; 
				}
				return mapstring;
			}

			/// <summary>
			/// Retrieves the map string for a plate, using a specified calibration.
			/// </summary>
			/// <param name="brickid">the brick to which the plate belongs.</param>
			/// <param name="plateid">the plate for which the map string is sought.</param>
			/// <param name="calibration">the Id of the calibration operation to be used.</param>
			/// <param name="conn">the DB connection to be used.</param>
			/// <param name="trans">the DB transaction in which the command is to be executed.</param>
			/// <returns>the map string for the plate.</returns>
			public static string GetMapString(long brickid, long plateid, long calibration, OperaDbConnection conn, OperaDbTransaction trans)
			{
				System.Globalization.CultureInfo InvC = System.Globalization.CultureInfo.InvariantCulture;
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT MINX - ZEROX, MAXX - ZEROX, MINY - ZEROY, MAXY - ZEROY FROM TB_EVENTBRICKS WHERE ID = " + brickid, conn, trans).Fill(ds);
				SySal.DAQSystem.Scanning.IntercalibrationInfo intercal = new SySal.DAQSystem.Scanning.IntercalibrationInfo();
				double MinX = Convert.ToDouble(ds.Tables[0].Rows[0][0]);
				double MaxX = Convert.ToDouble(ds.Tables[0].Rows[0][1]);
				double MinY = Convert.ToDouble(ds.Tables[0].Rows[0][2]);
				double MaxY = Convert.ToDouble(ds.Tables[0].Rows[0][3]);
				intercal.RX = (MinX + MaxX) * 0.5;
				intercal.RY = (MinY + MaxY) * 0.5;

                string markset = "";
				ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT MAPXX, MAPXY, MAPYX, MAPYY, MAPDX, MAPDY, MARKSETS FROM TB_PLATE_CALIBRATIONS WHERE ID_EVENTBRICK = " + brickid + " AND ID_PLATE = " + plateid + " AND ID_PROCESSOPERATION = " + calibration, conn, trans).Fill(ds);
				intercal.MXX = Convert.ToDouble(ds.Tables[0].Rows[0][0]);
				intercal.MXY = Convert.ToDouble(ds.Tables[0].Rows[0][1]);
				intercal.MYX = Convert.ToDouble(ds.Tables[0].Rows[0][2]);
				intercal.MYY = Convert.ToDouble(ds.Tables[0].Rows[0][3]);
				intercal.TX = Convert.ToDouble(ds.Tables[0].Rows[0][4]);
				intercal.TY = Convert.ToDouble(ds.Tables[0].Rows[0][5]);
				intercal.TZ = 0.0;
                markset = ds.Tables[0].Rows[0][6].ToString();
                if (markset.Length > 1) throw new Exception("Calibrations of multiple mark sets are not supported yet. Combination '" + markset + "' requested.");

				ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID, ID_MARK, POSX, POSY, SIDE FROM TB_TEMPLATEMARKSETS WHERE (ID_EVENTBRICK = " + brickid + ") AND INSTR(SHAPE,'" + markset + "') > 0 ORDER BY ID_MARK ASC", conn, trans).Fill(ds);
				Mark [] MarkSet = new Mark[ds.Tables[0].Rows.Count];
				string mapstring = ((markset == "L") ? "mapX" : "mapext: ") + brickid + " " + plateid + " 0 0; " + MarkSet.Length + " " + MinX + " " + MinY + " " + MaxX + " " + MaxY;
				int i;
				for (i = 0; i < ds.Tables[0].Rows.Count; i++)
				{
					MarkSet[i].IdTemplateMark = SySal.OperaDb.Convert.ToInt64(ds.Tables[0].Rows[i][0]);
					MarkSet[i].MarkId = SySal.OperaDb.Convert.ToInt32(ds.Tables[0].Rows[i][1]);
					MarkSet[i].TemplateX = SySal.OperaDb.Convert.ToDouble(ds.Tables[0].Rows[i][2]);
					MarkSet[i].TemplateY = SySal.OperaDb.Convert.ToDouble(ds.Tables[0].Rows[i][3]);
					MarkSet[i].X = intercal.MXX * (MarkSet[i].TemplateX - intercal.RX) + intercal.MXY * (MarkSet[i].TemplateY - intercal.RY) + intercal.TX + intercal.RX;
					MarkSet[i].Y = intercal.MYX * (MarkSet[i].TemplateX - intercal.RX) + intercal.MYY * (MarkSet[i].TemplateY - intercal.RY) + intercal.TY + intercal.RY;
					MarkSet[i].Side = SySal.OperaDb.Convert.ToInt32(ds.Tables[0].Rows[i][4]);
					mapstring += "; " + MarkSet[i].MarkId + " " + MarkSet[i].X + " " + MarkSet[i].Y + " " + MarkSet[i].TemplateX + " " + MarkSet[i].TemplateY + " 1 1 " + MarkSet[i].Side; 
				}
				return mapstring;
			}
		}
	}

	namespace ComputingInfrastructure
	{
		/// <summary>
		/// Helps creating and querying process operations.
		/// </summary>
		public class ProcessOperation
		{
			/// <summary>
			/// The member on which DB_Id relies. Can be accessed by derived classes.
			/// </summary>
			protected long m_DB_Id;
			/// <summary>
			/// Opera DB identifier of the operation.
			/// </summary>
			public long DB_Id { get { return m_DB_Id; } }
			/// <summary>
			/// The member on which ProgramSettings_Id relies. Can be accessed by derived classes.
			/// </summary>
			protected long m_ProgramSettings_Id;
			/// <summary>
			/// Opera DB identifier of the program settings for this operation.
			/// </summary>
			public long ProgramSettings_Id { get { return m_ProgramSettings_Id; } }
			/// <summary>
			/// The member on which DriverLevel relies. Can be accessed by derived classes.
			/// </summary>
			protected SySal.DAQSystem.Drivers.DriverType m_DriverLevel;
			/// <summary>
			/// Level of the driver to be executed.
			/// </summary>
			public SySal.DAQSystem.Drivers.DriverType DriverLevel { get { return m_DriverLevel; } }
			/// <summary>
			/// The member on which UsesTemplateMarks relies. Can be accessed by derived classes.
			/// </summary>
			protected bool m_UsesTemplateMarks;
			/// <summary>
			/// Tells whether this operation uses template marks. It is meaningful only for Scanning drivers.
			/// </summary>
			public bool UsesTemplateMarks { get { return m_UsesTemplateMarks; } }
			/// <summary>
			/// The member on which Requester_Id relies. Can be accessed by derived classes.
			/// </summary>
			protected long m_Requester_Id;
			/// <summary>
			/// Opera DB identifier of the user that requested this operation.
			/// </summary>
			public long Requester_Id { get { return m_Requester_Id; } }
			/// <summary>
			/// The member on which Machine_Id relies. Can be accessed by derived classes.
			/// </summary>
			protected long m_Machine_Id;
			/// <summary>
			/// Opera DB identifier of the machine on which the operation was scheduled.
			/// </summary>
			public long Machine_Id { get { return m_Machine_Id; } }
			/// <summary>
			/// The member on which Parent_Id relies. Can be accessed by derived classes.
			/// </summary>
			protected long m_Parent_Id;
			/// <summary>
			/// Identifier of the process operation that spawned this process operation.
			/// If this operation does not depend on any process operation, Parent_Id is a non-positive number.
			/// </summary>
			public long Parent_Id { get { return m_Parent_Id; } }
			/// <summary>
			/// The member on which EventBrick_Id relies. Can be accessed by derived classes.
			/// </summary>
			protected long m_EventBrick_Id;
			/// <summary>
			/// The Opera DB identifier of the Event/Brick. Zero if meaningless for this operation.
			/// </summary>
			public long EventBrick_Id { get { return m_EventBrick_Id; } }
			/// <summary>
			/// The member of which Plate_Id relies. Can be accessed by derived classes.
			/// </summary>
			protected long m_Plate_Id;
			/// <summary>
			/// The Opera DB identifier of the plate. Zero if meaningless for this operation.
			/// </summary>
			public long Plate_Id { get { return m_Plate_Id; } }
			/// <summary>
			/// The member on which IsStarted relies. Can be accessed by derived classes.
			/// </summary>
			public bool m_IsStarted = false;
			/// <summary>
			/// True if the operation started, false otherwise.
			/// </summary>
			public bool IsStarted { get { return m_IsStarted; } }
			/// <summary>
			/// The member on which m_StartTime relies. Can be accessed by derived classes.
			/// </summary>
			protected System.DateTime m_StartTime;
			/// <summary>
			/// The start time of the operation.
			/// </summary>
			public System.DateTime StartTime 
			{ 
				get 
				{ 
					if (!m_IsStarted) throw new Exception("Operation has not started yet.");
					return m_StartTime; 				
				} 
			}
			/// <summary>
			/// The member on which IsComplete relies. Can be accessed by derived classes.
			/// </summary>
			protected bool m_IsComplete = false;
			/// <summary>
			/// True if the operation completed, false if it is still going on.
			/// </summary>
			public bool IsComplete { get { return m_IsComplete; } }
			/// <summary>
			/// The member on which m_FinishTime relies. Can be accessed by derived classes.
			/// </summary>			
			protected System.DateTime m_FinishTime;			
			/// <summary>
			/// The finish time of the operation. 
			/// An exception is thrown if the operation is not complete.
			/// </summary>
			public System.DateTime FinishTime 
			{ 
				get 
				{ 
					if (!m_IsComplete) throw new Exception("Operation is still in progress.");
					return m_FinishTime; 
				} 
			}
			/// <summary>
			/// The member on which m_Success relies. Can be accessed by derived classes.
			/// </summary>
			protected bool m_Success;
			/// <summary>
			/// True if the operation completed successfully, false if it failed. 
			/// An exception is thrown if the operation is not complete.
			/// </summary>
			public bool Success 
			{ 
				get 
				{ 
					if (!m_IsComplete) throw new Exception("Operation is still in progress.");
					return m_Success; 
				} 
			}

			/// <summary>
			/// Starts a new process operation that does not require brick/plate information.
			/// </summary>
			/// <param name="idparent">process operation that spawned this operation. Zero or negative means NULL.</param>
			/// <param name="idmachine">machine on which the operation is to be performed.</param>
			/// <param name="idprogramsettings">program settings for this operation.</param>
			/// <param name="iduser">the user that requests the new operation.</param>
			/// <param name="notes">notes to be added to the process operation. Can be null.</param>
			/// <param name="conn">the DB connection to be used.</param>
			/// <param name="trans">the DB transaction to be used.</param>
			/// <returns>the id of the new operation.</returns>
			public static long Start(long idparent, long idmachine, long idprogramsettings, long iduser, string notes, OperaDbConnection conn, OperaDbTransaction trans)
			{
				OracleConnection Conn = (OracleConnection)conn.Conn;
				Oracle.ManagedDataAccess.Client.OracleCommand cmdproc = new Oracle.ManagedDataAccess.Client.OracleCommand("CALL PC_ADD_PROC_OPERATION(" + idmachine + ", " + idprogramsettings + ", " + iduser + ", " + ((idparent <= 0) ? "NULL" : idparent.ToString()) + ", TO_TIMESTAMP('" + OperaDbConnection.ToTimeFormat(System.DateTime.Now) + "' ," + OperaDbConnection.TimeFormat + "), :notes, :newid)", Conn);
				cmdproc.Parameters.Add("notes", Oracle.ManagedDataAccess.Client.OracleDbType.Varchar2, System.Data.ParameterDirection.Input);
				if ((notes == null) || (notes.Trim().Length == 0)) cmdproc.Parameters[0].Value = System.DBNull.Value;
				else cmdproc.Parameters[0].Value = notes.Trim();
				cmdproc.Parameters.Add("newid", Oracle.ManagedDataAccess.Client.OracleDbType.Int64, System.Data.ParameterDirection.Output);
				cmdproc.ExecuteNonQuery();
				return Convert.ToInt64(cmdproc.Parameters[1].Value);
			}

			/// <summary>
			/// Starts a new process operation that does not require brick/plate information. Checks user privileges and associates a token to the new operation.
			/// </summary>
			/// <param name="idparent">process operation that spawned this operation. Zero or negative means NULL.</param>
			/// <param name="idmachine">machine on which the operation is to be performed.</param>
			/// <param name="idprogramsettings">program settings for this operation.</param>
			/// <param name="username">the name of the user that attempts to start the operation. Ignored if the operation is a child of another operation (idparent nonzero).</param>
			/// <param name="password">the password of the user that attempts to start the operation. Ignored if the operation is a child of another operation (idparent nonzero).</param>
			/// <param name="token">on completion, this output parameter is the token assigned to the operation.</param>
			/// <param name="iduser">on completion, this output parameter is the user that requests the new operation.</param>
			/// <param name="notes">notes to be added to the process operation. Can be null.</param>
			/// <param name="conn">the DB connection to be used.</param>
			/// <param name="trans">the DB transaction to be used.</param>
			/// <returns>the id of the new operation.</returns>
			/// <remarks>The user credentials are relevant only for topmost operations (parentless operations). For children operations, the username/password pair is ignored, and security checks are performed only using the token of the parent operation (which the children will inherit).</remarks>
			public static long StartTokenized(long idparent, long idmachine, long idprogramsettings, string username, string password, out string token, out long iduser, string notes, OperaDbConnection conn, OperaDbTransaction trans)
			{
				OracleConnection Conn = (OracleConnection)conn.Conn;
				Oracle.ManagedDataAccess.Client.OracleCommand cmdproc = new Oracle.ManagedDataAccess.Client.OracleCommand("CALL LP_ADD_PROC_OPERATION(" + idmachine + ", " + idprogramsettings + ", :i_usr, :i_pwd, :o_token, :o_uid, " + ((idparent <= 0) ? "NULL" : idparent.ToString()) + ", TO_TIMESTAMP('" + OperaDbConnection.ToTimeFormat(System.DateTime.Now) + "' ," + OperaDbConnection.TimeFormat + "), :notes, :newid)", Conn);
				cmdproc.Parameters.Add("i_usr", OracleDbType.Varchar2, ParameterDirection.Input).Value = username;
				cmdproc.Parameters.Add("i_pwd", OracleDbType.Varchar2, ParameterDirection.Input).Value = password;
				OracleParameter o_token = new OracleParameter("o_token", OracleDbType.Varchar2, 256);
				o_token.Direction = ParameterDirection.Output;
				cmdproc.Parameters.Add(o_token);
				OracleParameter o_uid = new OracleParameter("o_uid", OracleDbType.Int64);
				o_uid.Direction = ParameterDirection.Output;
				cmdproc.Parameters.Add(o_uid);
				OracleParameter o_notes = cmdproc.Parameters.Add("notes", Oracle.ManagedDataAccess.Client.OracleDbType.Varchar2, System.Data.ParameterDirection.Input);
				if ((notes == null) || (notes.Trim().Length == 0)) o_notes.Value = System.DBNull.Value;
				else o_notes.Value = notes.Trim();
				OracleParameter newid = new OracleParameter("newid", OracleDbType.Int64);
				newid.Direction = ParameterDirection.Output;
				cmdproc.Parameters.Add(newid);
				cmdproc.ExecuteNonQuery();
				token = o_token.Value.ToString();
				iduser = Convert.ToInt64(o_uid.Value);
				return Convert.ToInt64(newid.Value);
			}

			/// <summary>
			/// Starts a new process operation that requires brick information.
			/// </summary>
			/// <param name="idparent">process operation that spawned this operation. Zero or negative means NULL.</param>
			/// <param name="idmachine">machine on which the operation is to be performed.</param>
			/// <param name="idprogramsettings">program settings for this operation.</param>
			/// <param name="iduser">the user that requests the new operation.</param>
			/// <param name="idbrick">the brick on which the operation is to be performed.</param>
			/// <param name="notes">notes to be added to the process operation. Can be null.</param>
			/// <param name="conn">the DB connection to be used.</param>
			/// <param name="trans">the DB transaction to be used.</param>
			/// <returns>the id of the new operation.</returns>
			public static long Start(long idparent, long idmachine, long idprogramsettings, long iduser, long idbrick, string notes, OperaDbConnection conn, OperaDbTransaction trans)
			{
				OracleConnection Conn = (OracleConnection)conn.Conn;
				Oracle.ManagedDataAccess.Client.OracleCommand cmdproc = new Oracle.ManagedDataAccess.Client.OracleCommand("CALL PC_ADD_PROC_OPERATION_BRICK(" + idmachine + ", " + idprogramsettings + ", " + iduser + ", " + idbrick + ", " + ((idparent <= 0) ? "NULL" : idparent.ToString()) + ", TO_TIMESTAMP('" + OperaDbConnection.ToTimeFormat(System.DateTime.Now) + "' ," + OperaDbConnection.TimeFormat + "), :notes, :newid)", Conn);
				cmdproc.Parameters.Add("notes", Oracle.ManagedDataAccess.Client.OracleDbType.Varchar2, System.Data.ParameterDirection.Input);
				if ((notes == null) || (notes.Trim().Length == 0)) cmdproc.Parameters[0].Value = System.DBNull.Value;
				else cmdproc.Parameters[0].Value = notes.Trim();
				cmdproc.Parameters.Add("newid", Oracle.ManagedDataAccess.Client.OracleDbType.Int64, System.Data.ParameterDirection.Output);
				cmdproc.ExecuteNonQuery();
				return Convert.ToInt64(cmdproc.Parameters[1].Value);
			}

			/// <summary>
			/// Starts a new process operation that requires brick information. Checks user privileges and associates a token to the new operation.
			/// </summary>
			/// <param name="idparent">process operation that spawned this operation. Zero or negative means NULL.</param>
			/// <param name="idmachine">machine on which the operation is to be performed.</param>
			/// <param name="idprogramsettings">program settings for this operation.</param>
			/// <param name="username">the name of the user that attempts to start the operation. Ignored if the operation is a child of another operation (idparent nonzero).</param>
			/// <param name="password">the password of the user that attempts to start the operation. Ignored if the operation is a child of another operation (idparent nonzero).</param>
			/// <param name="token">on completion, this output parameter is the token assigned to the operation.</param>
			/// <param name="iduser">on completion, this output parameter is the user that requests the new operation.</param>
			/// <param name="idbrick">the brick on which the operation is to be performed.</param>
			/// <param name="notes">notes to be added to the process operation. Can be null.</param>
			/// <param name="conn">the DB connection to be used.</param>
			/// <param name="trans">the DB transaction to be used.</param>
			/// <returns>the id of the new operation.</returns>
			/// <remarks>The user credentials are relevant only for topmost operations (parentless operations). For children operations, the username/password pair is ignored, and security checks are performed only using the token of the parent operation (which the children will inherit).</remarks>
			public static long StartTokenized(long idparent, long idmachine, long idprogramsettings, string username, string password, out string token, out long iduser, long idbrick, string notes, OperaDbConnection conn, OperaDbTransaction trans)
			{
				OracleConnection Conn = (OracleConnection)conn.Conn;
				Oracle.ManagedDataAccess.Client.OracleCommand cmdproc = new Oracle.ManagedDataAccess.Client.OracleCommand("CALL LP_ADD_PROC_OPERATION_BRICK(" + idmachine + ", " + idprogramsettings + ", :i_usr, :i_pwd, :o_token, :o_uid, " + idbrick + ", " + ((idparent <= 0) ? "NULL" : idparent.ToString()) + ", TO_TIMESTAMP('" + OperaDbConnection.ToTimeFormat(System.DateTime.Now) + "' ," + OperaDbConnection.TimeFormat + "), :notes, :newid)", Conn);
				cmdproc.Parameters.Add("i_usr", OracleDbType.Varchar2, ParameterDirection.Input).Value = username;
				cmdproc.Parameters.Add("i_pwd", OracleDbType.Varchar2, ParameterDirection.Input).Value = password;
				OracleParameter o_token = new OracleParameter("o_token", OracleDbType.Varchar2, 256);
				o_token.Direction = ParameterDirection.Output;
				cmdproc.Parameters.Add(o_token);
				OracleParameter o_uid = new OracleParameter("o_uid", OracleDbType.Int64);
				o_uid.Direction = ParameterDirection.Output;
				cmdproc.Parameters.Add(o_uid);				
				OracleParameter o_notes = cmdproc.Parameters.Add("notes", Oracle.ManagedDataAccess.Client.OracleDbType.Varchar2, System.Data.ParameterDirection.Input);
				if ((notes == null) || (notes.Trim().Length == 0)) o_notes.Value = System.DBNull.Value;
				else o_notes.Value = notes.Trim();
				OracleParameter newid = new OracleParameter("newid", OracleDbType.Int64);
				newid.Direction = ParameterDirection.Output;
				cmdproc.Parameters.Add(newid);
				cmdproc.ExecuteNonQuery();
				token = o_token.Value.ToString();
				iduser = Convert.ToInt64(o_uid.Value);
				return Convert.ToInt64(newid.Value);
			}

			/// <summary>
			/// Starts a new process operation that requires plate information.
			/// </summary>
			/// <param name="idparent">process operation that spawned this operation. Zero or negative means NULL.</param>
			/// <param name="idcalibration">calibration used by this operation. Zero or negative means NULL.</param>
			/// <param name="idmachine">machine on which the operation is to be performed.</param>
			/// <param name="idprogramsettings">program settings for this operation.</param>
			/// <param name="iduser">the user that requests the new operation.</param>
			/// <param name="idbrick">the brick on which the operation is to be performed.</param>
			/// <param name="idplate">the plate on which the operation is to be performed.</param>
			/// <param name="notes">notes to be added to the process operation. Can be null.</param>
			/// <param name="conn">the DB connection to be used.</param>
			/// <param name="trans">the DB transaction to be used.</param>
			/// <returns>the id of the new operation.</returns>
			public static long Start(long idparent, long idcalibration, long idmachine, long idprogramsettings, long iduser, long idbrick, long idplate, string notes, OperaDbConnection conn, OperaDbTransaction trans)
			{
				OracleConnection Conn = (OracleConnection)conn.Conn;
				Oracle.ManagedDataAccess.Client.OracleCommand cmdproc = new Oracle.ManagedDataAccess.Client.OracleCommand("CALL PC_ADD_PROC_OPERATION_PLATE(" + idmachine + ", " + idprogramsettings + ", " + iduser + ", " + idbrick + ", " + idplate + ", " + ((idparent <= 0) ? "NULL" : idparent.ToString()) + ", " + ((idcalibration <= 0) ? "NULL" : idcalibration.ToString()) + ", TO_TIMESTAMP('" + OperaDbConnection.ToTimeFormat(System.DateTime.Now) + "' ," + OperaDbConnection.TimeFormat + "), :notes, :newid)", Conn);
				cmdproc.Parameters.Add("notes", Oracle.ManagedDataAccess.Client.OracleDbType.Varchar2, System.Data.ParameterDirection.Input);
				if ((notes == null) || (notes.Trim().Length == 0)) cmdproc.Parameters[0].Value = System.DBNull.Value;
				else cmdproc.Parameters[0].Value = notes.Trim();
				cmdproc.Parameters.Add("newid", Oracle.ManagedDataAccess.Client.OracleDbType.Int64, System.Data.ParameterDirection.Output);
				cmdproc.ExecuteNonQuery();
				return Convert.ToInt64(cmdproc.Parameters[1].Value);
			}

			/// <summary>
			/// Starts a new process operation that requires plate information. Checks user privileges and associates a token to the new operation.
			/// </summary>
			/// <param name="idparent">process operation that spawned this operation. Zero or negative means NULL.</param>
			/// <param name="idcalibration">calibration used by this operation. Zero or negative means NULL.</param>
			/// <param name="idmachine">machine on which the operation is to be performed.</param>
			/// <param name="idprogramsettings">program settings for this operation.</param>
			/// <param name="username">the name of the user that attempts to start the operation. Ignored if the operation is a child of another operation (idparent nonzero).</param>
			/// <param name="password">the password of the user that attempts to start the operation. Ignored if the operation is a child of another operation (idparent nonzero).</param>
			/// <param name="token">on completion, this output parameter is the token assigned to the operation.</param>
			/// <param name="iduser">on completion, this output parameter is the user that requests the new operation.</param>
			/// <param name="idbrick">the brick on which the operation is to be performed.</param>
			/// <param name="idplate">the plate on which the operation is to be performed.</param>
			/// <param name="notes">notes to be added to the process operation. Can be null.</param>
			/// <param name="conn">the DB connection to be used.</param>
			/// <param name="trans">the DB transaction to be used.</param>
			/// <returns>the id of the new operation.</returns>
			/// <remarks>The user credentials are relevant only for topmost operations (parentless operations). For children operations, the username/password pair is ignored, and security checks are performed only using the token of the parent operation (which the children will inherit).</remarks>
			public static long StartTokenized(long idparent, long idcalibration, long idmachine, long idprogramsettings, string username, string password, out string token, out long iduser, long idbrick, long idplate, string notes, OperaDbConnection conn, OperaDbTransaction trans)
			{
				OracleConnection Conn = (OracleConnection)conn.Conn;
				Oracle.ManagedDataAccess.Client.OracleCommand cmdproc = new Oracle.ManagedDataAccess.Client.OracleCommand("CALL LP_ADD_PROC_OPERATION_PLATE(" + idmachine + ", " + idprogramsettings + ", :i_usr, :i_pwd, :o_token, :o_uid, " + idbrick + ", " + idplate + ", " + ((idparent <= 0) ? "NULL" : idparent.ToString()) + ", " + ((idcalibration <= 0) ? "NULL" : idcalibration.ToString()) + ", TO_TIMESTAMP('" + OperaDbConnection.ToTimeFormat(System.DateTime.Now) + "' ," + OperaDbConnection.TimeFormat + "), :notes, :newid)", Conn);
				cmdproc.Parameters.Add("i_usr", OracleDbType.Varchar2, ParameterDirection.Input).Value = username;
				cmdproc.Parameters.Add("i_pwd", OracleDbType.Varchar2, ParameterDirection.Input).Value = password;
				OracleParameter o_token = new OracleParameter("o_token", OracleDbType.Varchar2, 256);
				o_token.Direction = ParameterDirection.Output;
				cmdproc.Parameters.Add(o_token);
				OracleParameter o_uid = new OracleParameter("o_uid", OracleDbType.Int64);
				o_uid.Direction = ParameterDirection.Output;
				cmdproc.Parameters.Add(o_uid);
				OracleParameter o_notes = cmdproc.Parameters.Add("notes", Oracle.ManagedDataAccess.Client.OracleDbType.Varchar2, System.Data.ParameterDirection.Input);
				if ((notes == null) || (notes.Trim().Length == 0)) o_notes.Value = System.DBNull.Value;
				else o_notes.Value = notes.Trim();
				OracleParameter newid = new OracleParameter("newid", OracleDbType.Int64);
				newid.Direction = ParameterDirection.Output;
				cmdproc.Parameters.Add(newid);
				cmdproc.ExecuteNonQuery();
				token = o_token.Value.ToString();
				iduser = Convert.ToInt64(o_uid.Value);
				return Convert.ToInt64(newid.Value);
			}

			/// <summary>
			/// Completes a process operation.
			/// </summary>
			/// <param name="id">the process operation to be completed.</param>
			/// <param name="success">true if the operation completed successfully, false if it failed.</param>
			/// <param name="conn">DB connection to be used.</param>
			/// <param name="trans">DB transaction to be used.</param>
			public static void End(long id, bool success, OperaDbConnection conn, OperaDbTransaction trans)
			{
				OracleConnection Conn = (OracleConnection)conn.Conn;
				if (success) new OracleCommand("CALL PC_SUCCESS_OPERATION(" + id + ", TO_TIMESTAMP('" + OperaDbConnection.ToTimeFormat(System.DateTime.Now) + "' ," + OperaDbConnection.TimeFormat + "))", Conn).ExecuteNonQuery();
				else new OracleCommand("CALL PC_FAIL_OPERATION(" + id + ", TO_TIMESTAMP('" + OperaDbConnection.ToTimeFormat(System.DateTime.Now) + "' ," + OperaDbConnection.TimeFormat + "))", Conn).ExecuteNonQuery();
			}

			/// <summary>
			/// Completes a process operation. The reference count of the associated token is updated, and the token is released if no more references exist.
			/// </summary>
			/// <param name="id">the process operation to be completed.</param>
			/// <param name="success">true if the operation completed successfully, false if it failed.</param>
			/// <param name="conn">DB connection to be used.</param>
			/// <param name="trans">DB transaction to be used.</param>
			public static void EndTokenized(long id, bool success, OperaDbConnection conn, OperaDbTransaction trans)
			{
				OracleConnection Conn = (OracleConnection)conn.Conn;
				if (success) new OracleCommand("CALL LP_SUCCESS_OPERATION(" + id + ", TO_TIMESTAMP('" + OperaDbConnection.ToTimeFormat(System.DateTime.Now) + "' ," + OperaDbConnection.TimeFormat + "))", Conn).ExecuteNonQuery();
				else new OracleCommand("CALL LP_FAIL_OPERATION(" + id + ", TO_TIMESTAMP('" + OperaDbConnection.ToTimeFormat(System.DateTime.Now) + "' ," + OperaDbConnection.TimeFormat + "))", Conn).ExecuteNonQuery();
			}

			/// <summary>
			/// Completes a process operation.
			/// </summary>
			/// <param name="success">true if the operation completed successfully, false if it failed.</param>
			/// <param name="conn">DB connection to be used.</param>
			/// <param name="trans">DB transaction to be used.</param>
			public void End(bool success, OperaDbConnection conn, OperaDbTransaction trans)
			{
				End(m_DB_Id, success, conn, trans);
				m_IsComplete = true;
				m_Success = success;
				m_FinishTime = (System.DateTime)new OperaDbCommand("SELECT FINISHTIME FROM TB_PROC_OPERATIONS WHERE (ID = " + m_DB_Id + ")", conn, trans).ExecuteScalar();
			}

			protected static OperaDbCommand s_StatusCmd = new OperaDbCommand("SELECT /*+INDEX (TB_PROC_OPERATIONS PK_PROC_OPERATIONS)*/ SUCCESS FROM TB_PROC_OPERATIONS WHERE (ID = :id)");
			/// <summary>
			/// Returns the completion status of the specified operation.
			/// </summary>
			/// <param name="id">id of the operation to be queried.</param>
			/// <param name="conn">DB connection to be used.</param>
			/// <param name="trans">DB transaction to be used.</param>
			/// <returns>the completion status of the specified operation.</returns>
			public static SySal.DAQSystem.Drivers.Status Status(long id, OperaDbConnection conn, OperaDbTransaction trans)
			{				
				lock (s_StatusCmd)
				{
					if (s_StatusCmd.Connection == null) 
					{
						s_StatusCmd.Connection = conn;
						s_StatusCmd.Parameters.Add("id", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
					}
					else s_StatusCmd.Connection = conn;
					try
					{					
						s_StatusCmd.Parameters[0].Value = id;
						string o = s_StatusCmd.ExecuteScalar().ToString();
						if (o == "R") return SySal.DAQSystem.Drivers.Status.Running;
						if (o == "N") return SySal.DAQSystem.Drivers.Status.Failed;
						if (o == "Y") return SySal.DAQSystem.Drivers.Status.Completed;
					}
					catch (Exception)
					{
						return SySal.DAQSystem.Drivers.Status.Unknown;
					}
					return SySal.DAQSystem.Drivers.Status.Completed;
				}
			}

			/// <summary>
			/// Protected constructor. Prevents users from creating an instance of ProcessOperation without initializing it. Is implicitly called by constructors in derived classes.
			/// </summary>
			protected ProcessOperation() {}

			/// <summary>
			/// Reads a process operation from the Opera DB.
			/// </summary>
			/// <param name="id">the Opera DB identifier of the ProcessOperation to be read.</param>
			/// <param name="conn">DB connection to be used.</param>
			/// <param name="trans">DB transaction to be used.</param>
			public ProcessOperation(long id, OperaDbConnection conn, OperaDbTransaction trans)
			{
				System.Data.DataSet ds = new System.Data.DataSet();
				SySal.OperaDb.OperaDbDataAdapter da = new SySal.OperaDb.OperaDbDataAdapter("SELECT (ID_PROGRAMSETTINGS, ID_REQUESTER, ID_MACHINE, ID_PARENT_OPERATION, DRIVERLEVEL, TEMPLATEMARKS, ID_EVENTBRICK, ID_PLATE, STARTTIME, FINISHTIME, SUCCESS) FROM TB_PROC_OPERATIONS WHERE (ID = " + id + ")", conn, trans);
				da.Fill(ds);
				m_DB_Id = id;
				System.Data.DataRow dr = ds.Tables[0].Rows[0];
				m_ProgramSettings_Id = Convert.ToInt64(dr[0]);
				m_Requester_Id = Convert.ToInt64(dr[1]);
				m_Machine_Id = Convert.ToInt64(dr[2]);
				try
				{
					m_Parent_Id = Convert.ToInt64(dr[3]);
				}
				catch (Exception)
				{
					m_Parent_Id = 0;
				}
				m_DriverLevel = (SySal.DAQSystem.Drivers.DriverType)Convert.ToInt64(dr[4]);
				m_UsesTemplateMarks = Convert.ToInt64(dr[5]) != 1;
				m_EventBrick_Id = Convert.ToInt64(dr[6]);
				m_Plate_Id = Convert.ToInt64(dr[7]);
				m_StartTime = (System.DateTime)dr[8];
				m_IsStarted = true;
				try
				{
					m_FinishTime = (System.DateTime)dr[5];
					m_Success = (dr[6].ToString() == "Y");
					m_IsComplete = true;
				}
				catch (Exception)
				{
					m_IsComplete = false;
					m_Success = false;
				}
			}
		}

		/// <summary>
		/// A machine entry in the DB.
		/// </summary>
		public class Machine
		{
			/// <summary>
			/// Protected data member on which the Address property relies. Can be accessed from derived classes.
			/// </summary>
			protected string m_Address;
			/// <summary>
			/// Address of the user's office.
			/// </summary>
			public string Address { get { return m_Address; } }
			/// <summary>
			/// Protected data member on which the DB_Id property relies. Can be accessed from derived classes.
			/// </summary>
			protected long m_DB_Id;
			/// <summary>
			/// Opera DB identifier of the user entry.
			/// </summary>
			public long DB_Id { get { return m_DB_Id; } }
			/// <summary>
			/// Protected data member on which the DB_Id_Site property relies. Can be accessed from derived classes.
			/// </summary>
			protected long m_DB_Id_Site;
			/// <summary>
			/// Opera DB identifier of the site the user belongs to.
			/// </summary>
			public long DB_Id_Site { get { return m_DB_Id_Site; } }
			/// <summary>
			/// Protected data member on which the Name property relies. Can be accessed from derived classes.
			/// </summary>
			protected string m_Name;
			/// <summary>
			/// Friendly name of the machine.
			/// </summary>
			public string Name { get { return m_Name; } }
			/// <summary>
			/// Protected data member on which the IsScanningServer property relies. Can be accessed from derived classes.
			/// </summary>
			protected bool m_IsScanningServer;
			/// <summary>
			/// Tells whether the machine can run scanning tasks.
			/// </summary>
			public bool IsScanningServer { get { return m_IsScanningServer; } }
			/// <summary>
			/// Protected data member on which the IsBatchServer property relies. Can be accessed from derived classes.
			/// </summary>
			protected bool m_IsBatchServer;
			/// <summary>
			/// Tells whether the machine can control several scanning servers and manage scanning batches.
			/// </summary>
			public bool IsBatchServer { get { return m_IsBatchServer; } }
			/// <summary>
			/// Protected data member on which the IsDataProcessingServer property relies. Can be accessed from derived classes.
			/// </summary>
			protected bool m_IsDataProcessingServer;
			/// <summary>
			/// Tells whether the machine can perform batch data processing.
			/// </summary>
			public bool IsDataProcessingServer { get { return m_IsDataProcessingServer; } }
			/// <summary>
			/// Protected data member on which the IsDataBaseServer property relies. Can be accessed from derived classes.
			/// </summary>			
			protected bool m_IsDataBaseServer;
			/// <summary>
			/// Tells whether the machine hosts a local instance of the Opera DB.
			/// </summary>
			public bool IsDataBaseServer { get { return m_IsDataBaseServer; } }
			/// <summary>
			/// Protected data member on which the IsWebServer property relies. Can be accessed from derived classes.
			/// </summary>			
			protected bool m_IsWebServer;
			/// <summary>
			/// Tells whether the machine is a local Web server that can manage the scanning cluster.
			/// </summary>
			public bool IsWebServer { get { return m_IsWebServer; } }
			/// <summary>
			/// Protected constructor. Prevents users from creating instances of the Machine class without deriving it. Is implicitly called by constructors in derived classes.
			/// </summary>
			protected Machine() {}
			/// <summary>
			/// Reads a machine from an Opera DB.
			/// </summary>
			/// <param name="id">the DB identifier for the machine entry to be read.</param>
			/// <param name="conn">the DB connection to be used.</param>
			/// <param name="trans">the DB transaction to be used. Can be null.</param>
			public Machine(long id, OperaDbConnection conn, OperaDbTransaction trans)
			{
				//OracleDataAdapter da = new OracleDataAdapter("SELECT TB_MACHINES.NAME, TB_MACHINES.ADDRESS, TB_MACHINES.ISSCANNINGSERVER, TB_MACHINES.ISBATCHSERVER, TB_MACHINES.ISDATAPROCESSINGSERVER, TB_MACHINES.ISDATABASESERVER, TB_MACHINES.ISWEBSERVER, TB_MACHINES.ISLOCALMASTERCONTROLLER, TB_MACHINES.ID_SITE FROM TB_MACHINES WHERE (TB_MACHINES.ID = " + id + ")", (OracleConnection)conn.Conn);
				OracleDataAdapter da = new OracleDataAdapter("SELECT TB_MACHINES.NAME, TB_MACHINES.ADDRESS, TB_MACHINES.ISSCANNINGSERVER, TB_MACHINES.ISBATCHSERVER, TB_MACHINES.ISDATAPROCESSINGSERVER, TB_MACHINES.ISDATABASESERVER, TB_MACHINES.ISWEBSERVER, TB_MACHINES.ID_SITE FROM TB_MACHINES WHERE (TB_MACHINES.ID = " + id + ")", (OracleConnection)conn.Conn);
				DataSet ds = new DataSet();
				da.Fill(ds);
				DataRow dr = ds.Tables[0].Rows[0];
				m_Name = dr[0].ToString();
				m_Address = dr[1].ToString();
				m_IsScanningServer = (Convert.ToInt16(dr[2]) > 0);
				m_IsBatchServer = (Convert.ToInt16(dr[3]) > 0);
				m_IsDataProcessingServer = (Convert.ToInt16(dr[4]) > 0);
				m_IsDataBaseServer = (Convert.ToInt16(dr[5]) > 0);
				m_IsWebServer = (Convert.ToInt16(dr[6]) > 0);
				//m_IsLocalMasterController = (Convert.ToInt16(dr[7]) > 0);
				m_DB_Id_Site = Convert.ToInt64(dr[7]);
				m_DB_Id = id;
			}
			/// <summary>
			/// Adds a machine entry to an Opera DB.
			/// </summary>
			/// <param name="conn">the DB connection to be used.</param>
			/// <param name="trans">the DB transaction to be used.</param>
			/// <returns>the DB identifier that has been assigned to the machine.</returns>
			public long Add(OperaDbConnection conn, OperaDbTransaction trans)
			{
				/*OracleCommand cmd = new OracleCommand("INSERT INTO TB_MACHINES (NAME, ADDRESS, ISSCANNINGSERVER, ISBATCHSERVER, ISDATAPROCESSINGSERVER, ISDATABASESERVER, ISWEBSERVER, ISLOCALMASTERCONTROLLER, ID_SITE) VALUES ('" + 
					Name + "', '" + Address + "', " + (IsScanningServer ? "1" : "0") + ", " + (IsBatchServer ? "1" : "0") + ", " + (IsDataProcessingServer ? "1" : "0") + ", " + (IsDataBaseServer ? "1" : "0") + ", " + (IsWebServer ? "1" : "0") + ", " + (IsLocalMasterController ? "1" : "0") + ", " + DB_Id_Site + ") RETURNING TB_MACHINES.ID INTO :newid",
					(OracleConnection)conn.Conn);
				cmd.Parameters.Add("newid", OracleDbType.Int16); cmd.Parameters[0].Direction = ParameterDirection.Output;
				cmd.ExecuteNonQuery();
				return (m_DB_Id = Convert.ToInt64(cmd.Parameters[0].Value));
				*/

				//Using PC_ADD_MACHINE
				OracleCommand cmd= new OracleCommand("PC_ADD_MACHINE",(OracleConnection)conn.Conn);
				cmd.CommandType=CommandType.StoredProcedure;

				//setting parameters
				OracleParameter prm1=new OracleParameter("s_id",OracleDbType.Int64);
				prm1.Direction=ParameterDirection.Input;
				prm1.Value=DB_Id_Site;
				cmd.Parameters.Add(prm1);

				OracleParameter prm2=new OracleParameter("m_name",OracleDbType.Varchar2);
				prm2.Direction=ParameterDirection.Input;
				prm2.Value=Name;
				cmd.Parameters.Add(prm2);

				OracleParameter prm3=new OracleParameter("m_address",OracleDbType.Varchar2);
				prm3.Direction=ParameterDirection.Input;
				prm3.Value=Address;	
				cmd.Parameters.Add(prm3);

				OracleParameter prm4=new OracleParameter("m_scansrv",OracleDbType.Int16);
				prm4.Direction=ParameterDirection.Input;
				prm4.Value=(IsScanningServer ? "1" : "0");
				cmd.Parameters.Add(prm4);
								
				OracleParameter prm5=new OracleParameter("m_batchsrv",OracleDbType.Int16);
				prm5.Direction=ParameterDirection.Input;
				prm5.Value=(IsBatchServer ? "1" : "0");
				cmd.Parameters.Add(prm5);

				OracleParameter prm6=new OracleParameter("m_dataprocsrv",OracleDbType.Int16);
				prm6.Direction=ParameterDirection.Input;
				prm6.Value=(IsDataProcessingServer ? "1" : "0");
				cmd.Parameters.Add(prm6);
								
				OracleParameter prm7=new OracleParameter("m_websrv",OracleDbType.Int16);
				prm7.Direction=ParameterDirection.Input;
				prm7.Value=(IsWebServer ? "1" : "0");
				cmd.Parameters.Add(prm7);
				
				OracleParameter prm8=new OracleParameter("m_dbsrv",OracleDbType.Int16);
				prm8.Direction=ParameterDirection.Input;
				prm8.Value=(IsDataBaseServer ? "1" : "0");
				cmd.Parameters.Add(prm8);

				OracleParameter prm9=new OracleParameter("newid",OracleDbType.Int64);
				prm9.Direction=ParameterDirection.Output;
				cmd.Parameters.Add(prm9);

				cmd.ExecuteNonQuery();

				return (m_DB_Id = Convert.ToInt64(cmd.Parameters["newid"].Value));
			}
			/// <summary>
			/// Synchronizes a modified machine with its DB entry.
			/// The DB_Id must have been left unmodified.
			/// </summary>
			/// <param name="conn">the DB connection to be used.</param>
			/// <param name="trans">the DB transaction to be used. Cannot be null.</param>
			public void Synchronize(OperaDbConnection conn, OperaDbTransaction trans)
			{
				/*
								new OracleCommand("UPDATE TB_MACHINES SET NAME = '" + m_Name + "', ADDRESS = '" + m_Address + "', ISSCANNINGSERVER = " + (m_IsScanningServer ? "1" : "0") + ", ISBATCHSERVER = " + (m_IsBatchServer ? "1" : "0") + ", ISDATAPROCESSINGSERVER = " + (m_IsDataProcessingServer ? "1" : "0") + ", ISDATABASESERVER = " + (m_IsDataBaseServer ? "1" : "0") + ", ISWEBSERVER = " + (m_IsWebServer ? "1" : "0") + 
									", ISLOCALMASTERCONTROLLER = " + (m_IsLocalMasterController ? "1" : "0") + ", ID_SITE = " + m_DB_Id_Site + " WHERE(TB_MACHINES.ID = " + m_DB_Id + ")", (OracleConnection)conn.Conn).ExecuteNonQuery();
				*/
				//using PC_SET_MACHINE
				OracleCommand cmd= new OracleCommand("PC_SET_MACHINE",(OracleConnection)conn.Conn);
				cmd.CommandType=CommandType.StoredProcedure;

				//setting parameters
				OracleParameter prm1=new OracleParameter("m_id",OracleDbType.Int64);
				prm1.Direction=ParameterDirection.Input;
				prm1.Value=m_DB_Id;
				cmd.Parameters.Add(prm1);

				OracleParameter prm2=new OracleParameter("m_name",OracleDbType.Varchar2);
				prm2.Direction=ParameterDirection.Input;
				prm2.Value=Name;
				cmd.Parameters.Add(prm2);

				OracleParameter prm3=new OracleParameter("m_address",OracleDbType.Varchar2);
				prm3.Direction=ParameterDirection.Input;
				prm3.Value=Address;	
				cmd.Parameters.Add(prm3);

				OracleParameter prm4=new OracleParameter("m_scansrv",OracleDbType.Int16);
				prm4.Direction=ParameterDirection.Input;
				prm4.Value=(IsScanningServer ? "1" : "0");
				cmd.Parameters.Add(prm4);
								
				OracleParameter prm5=new OracleParameter("m_batchsrv",OracleDbType.Int16);
				prm5.Direction=ParameterDirection.Input;
				prm5.Value=(IsBatchServer ? "1" : "0");
				cmd.Parameters.Add(prm5);

				OracleParameter prm6=new OracleParameter("m_dataprocsrv",OracleDbType.Int16);
				prm6.Direction=ParameterDirection.Input;
				prm6.Value=(IsDataProcessingServer ? "1" : "0");
				cmd.Parameters.Add(prm6);
								
				OracleParameter prm7=new OracleParameter("m_websrv",OracleDbType.Int16);
				prm7.Direction=ParameterDirection.Input;
				prm7.Value=(IsWebServer ? "1" : "0");
				cmd.Parameters.Add(prm7);
				
				OracleParameter prm8=new OracleParameter("m_dbsrv",OracleDbType.Int16);
				prm8.Direction=ParameterDirection.Input;
				prm8.Value=(IsDataBaseServer ? "1" : "0");
				cmd.Parameters.Add(prm8);

				//executing query
				cmd.ExecuteNonQuery();

			}
			/// <summary>
			/// Deletes a machine from the Opera DB.
			/// </summary>
			/// <param name="id">the DB identifier of the machine to be deleted.</param>
			/// <param name="conn">the DB connection to be used.</param>
			/// <param name="trans">the DB transaction to be used. Cannot be null.</param>
			public static void Delete(long id, OperaDbConnection conn, OperaDbTransaction trans)
			{

				OracleTransaction tr = (trans == null) ? null : (OracleTransaction)trans.Trans;

				/*  			new OracleCommand("DELETE FROM TB_MACHINES WHERE (TB_MACHINES.ID = " + id + ")", (OracleConnection)conn.Conn).ExecuteNonQuery();
				*/
				//Using PC_DEL_MACHINE
				OracleCommand cmd= new OracleCommand("PC_DEL_MACHINE",(OracleConnection)conn.Conn);
				cmd.CommandType=CommandType.StoredProcedure;

				//setting parameters
				OracleParameter prm1=new OracleParameter("m_id",OracleDbType.Int64);
				prm1.Direction=ParameterDirection.Input;
				prm1.Value=id;
				cmd.Parameters.Add(prm1);

				cmd.ExecuteNonQuery();
			
			}
		}

		/// <summary>
		/// User permission values.
		/// </summary>
		public enum UserPermissionTriState 
		{ 
			/// <summary>
			/// The permission is denied.
			/// </summary>
			Deny, 
			/// <summary>
			/// The permission is granted.
			/// </summary>
			Grant, 
			/// <summary>
			/// The permission can be disregarded.
			/// This is only for permission checking, not for assignment.
			/// </summary>
			DontCare 
		}
		
		/// <summary>
		/// User permission designators.
		/// </summary>
		public enum UserPermissionDesignator 
		{ 
			/// <summary>
			/// Permission to request scanning tasks.
			/// </summary>
			Scan = 1,
			/// <summary>
			/// Permission to analyze data through the Web sites.
			/// </summary>
			WebAnalysis = 2,
			/// <summary>
			/// Permission to process data in batch mode by the data processing servers.
			/// </summary>
			ProcessData = 3,
			/// <summary>
			/// Permission to download large batches of data.
			/// </summary>
			DownloadData = 4,
			/// <summary>
			/// Permission to start brick processing.
			/// </summary>
			StartupProcess = 5,
			/// <summary>
			/// Permission to administer the site.
			/// </summary>
			Administer = 6
		}

		/// <summary>
		/// User permission.
		/// </summary>
		public struct UserPermission
		{
			/// <summary>
			/// DB identifier of the site this permission refers to.
			/// </summary>
			public long DB_Site_Id;

			/// <summary>
			/// Permission designator.
			/// </summary>
			public UserPermissionDesignator Designator;

			/// <summary>
			/// Permission value.
			/// </summary>
			public UserPermissionTriState Value;

			/// <summary>
			/// Retrieves an explanation of the permission designator.
			/// </summary>
			/// <param name="d">permission designator to be explained.</param>
			/// <returns>an explanation of the permission designator.</returns>
			public static string LongName(UserPermissionDesignator d)
			{
				switch (d)
				{
					case UserPermissionDesignator.Scan:				return "Start scanning tasks.";

					case UserPermissionDesignator.WebAnalysis:		return "Analyze data through Web servers.";

					case UserPermissionDesignator.ProcessData:		return "Process data by data processing servers.";

					case UserPermissionDesignator.DownloadData:		return "Download data.";

					case UserPermissionDesignator.StartupProcess:	return "Start brick processing.";

					case UserPermissionDesignator.Administer:		return "Administer the site.";
					
				}
				return "Unknown.";
			}

			/// <summary>
			/// Retrieves a short code for a permission designator.
			/// </summary>
			/// <param name="d">permission designator to be coded.</param>
			/// <returns>code for the permission designator.</returns>
			public static string ShortName(UserPermissionDesignator d)
			{
				switch (d)
				{
					case UserPermissionDesignator.Scan:				return "S";

					case UserPermissionDesignator.WebAnalysis:		return "W";

					case UserPermissionDesignator.ProcessData:		return "P";

					case UserPermissionDesignator.DownloadData:		return "D";

					case UserPermissionDesignator.StartupProcess:	return "B";

					case UserPermissionDesignator.Administer:		return "A";					
				}
				return "?";
			}

			internal static string DBColumnName(UserPermissionDesignator d)
			{
				switch (d)
				{
					case UserPermissionDesignator.Scan:				return "REQUESTSCAN";

					case UserPermissionDesignator.WebAnalysis:		return "REQUESTWEBANALYSIS";

					case UserPermissionDesignator.ProcessData:		return "REQUESTDATAPROCESSING";

					case UserPermissionDesignator.DownloadData:		return "REQUESTDATADOWNLOAD";

					case UserPermissionDesignator.StartupProcess:	return "REQUESTPROCESSSTARTUP";

					case UserPermissionDesignator.Administer:		return "ADMINISTER";
					
				}
				throw new Exception("Unknown permission designator.");
			}
		}

		/// <summary>
		/// A user of the Opera European Computing Infrastructure.
		/// </summary>
		public class User
		{
			/// <summary>
			/// Protected data member on which the UserName property relies. Can be accessed from derived classes.
			/// </summary>
			protected string m_UserName;
			/// <summary>
			/// Login name for the user.
			/// </summary>
			public string UserName { get { return m_UserName; } }
			/// <summary>
			/// Protected data member on which the Password property relies. Can be accessed from derived classes.
			/// </summary>
			protected string m_Password;
			/// <summary>
			/// Login password for the user.
			/// </summary>
			public string Password { get { return m_Password; } }
			/// <summary>
			/// Protected data member on which the Name property relies. Can be accessed from derived classes.
			/// </summary>
			protected string m_Name;
			/// <summary>
			/// The first name (given name) of the user.
			/// </summary>
			public string Name { get { return m_Name; } }
			/// <summary>
			/// Protected data member on which the Surname property relies. Can be accessed from derived classes.
			/// </summary>
			protected string m_Surname;
			/// <summary>
			/// The surname (family name) of the user.
			/// </summary>
			public string Surname { get { return m_Surname; } }
			/// <summary>
			/// Protected data member on which the Institution property relies. Can be accessed from derived classes.
			/// </summary>
			protected string m_Institution;
			/// <summary>
			/// The name of the institution the user belongs to.
			/// </summary>
			public string Institution { get { return m_Institution; } }
			/// <summary>
			/// Protected data member on which the EMailAddress property relies. Can be accessed from derived classes.
			/// </summary>
			protected string m_EMailAddress;
			/// <summary>
			/// E-Mail address of the user.
			/// </summary>
			public string EMailAddress { get { return m_EMailAddress; } }
			/// <summary>
			/// Protected data member on which the PhoneNumber property relies. Can be accessed from derived classes.
			/// </summary>
			protected string m_PhoneNumber;
			/// <summary>
			/// Phone number of the user.
			/// </summary>
			public string PhoneNumber { get { return m_PhoneNumber; } }
			/// <summary>
			/// Protected data member on which the Address property relies. Can be accessed from derived classes.
			/// </summary>
			protected string m_Address;
			/// <summary>
			/// Address of the user's office.
			/// </summary>
			public string Address { get { return m_Address; } }
			/// <summary>
			/// Protected data member on which the DB_Id property relies. Can be accessed from derived classes.
			/// </summary>
			protected long m_DB_Id;
			/// <summary>
			/// Opera DB identifier of the user entry.
			/// </summary>
			public long DB_Id { get { return m_DB_Id; } }
			/// <summary>
			/// Protected data member on which the DB_Id_Site property relies. Can be accessed from derived classes.
			/// </summary>
			protected long m_DB_Id_Site;
			/// <summary>
			/// Opera DB identifier of the site the user belongs to.
			/// </summary>
			public long DB_Id_Site { get { return m_DB_Id_Site; } }
			/// <summary>
			/// Permissions owned by the user.
			/// </summary>
			public UserPermission [] Permissions;
			/// <summary>
			/// Checks a user's login.
			/// </summary>
			/// <param name="username">login name of the user.</param>
			/// <param name="password">user login password.</param>
			/// <param name="conn">the DB connection to be used.</param>
			/// <param name="trans">the DB transaction to be used. Can be null.</param>
			/// <returns>the DB identifier for the user if the user exists; otherwise, an exception is thrown.</returns>
			public static long CheckLogin(string username, string password, OperaDbConnection conn, OperaDbTransaction trans)
			{
				OracleCommand cmd = new OracleCommand("CALL PC_CHECK_LOGIN(:uuser, :upwd, :usrid)", (OracleConnection)conn.Conn);
				cmd.Parameters.Add("uuser", Oracle.ManagedDataAccess.Client.OracleDbType.Varchar2, System.Data.ParameterDirection.Input).Value = username;
				cmd.Parameters.Add("upwd", Oracle.ManagedDataAccess.Client.OracleDbType.Varchar2, System.Data.ParameterDirection.Input).Value = password;
				cmd.Parameters.Add("usrid", Oracle.ManagedDataAccess.Client.OracleDbType.Int64, System.Data.ParameterDirection.Output);
				cmd.ExecuteNonQuery();
				return Convert.ToInt64(cmd.Parameters[2].Value);
			}


			/// <summary>
			/// Checks that a token owns specific privileges.
			/// </summary>
			/// <param name="token">the token to be checked.</param>
			/// <param name="rights">the set of access rights to be verified.</param>
			/// <param name="conn">the DB connection to be used.</param>
			/// <param name="trans">the DB transaction to be used.</param>
			/// <returns>true if all the specified privileges are owned, false otherwise.</returns>
			/// <remarks>
			/// <para><b>NOTICE: this method DOES NOT check that the site fields included in the <c>rights</c> actually match the ID_SITE of the accessed DB.</b> It is the responsibility of the caller to ensure that the requested access rights really refer to the accessed DB.</para>			
			/// </remarks>
			public static bool CheckTokenAccess(string token, UserPermission [] rights, OperaDbConnection conn, OperaDbTransaction trans)
			{
				int requestscan = 0;
				int requestwebanalysis = 0;				
				int requestdataprocessing = 0;
				int requestdatadownload = 0;
				int requestprocessstartup = 0;
				int administer = 0;
				foreach (UserPermission u in rights)
				{
					if (u.Value == UserPermissionTriState.Grant)
					{
						switch (u.Designator)
						{
							case UserPermissionDesignator.Administer: administer = 1; break;
							case UserPermissionDesignator.DownloadData: requestdatadownload = 1; break;
							case UserPermissionDesignator.ProcessData: requestdataprocessing = 1; break;
							case UserPermissionDesignator.Scan: requestscan = 1; break;
							case UserPermissionDesignator.StartupProcess: requestprocessstartup = 1; break;
							case UserPermissionDesignator.WebAnalysis: requestwebanalysis = 1; break;
							default: throw new Exception("Internal code inconsistency in UserPermissionDesignators!!!");
						}
					}
				}
				try
				{
                    OracleCommand cmd = new OracleCommand("CALL LP_CHECK_ACCESS('" + token + "', " + requestscan + ", " + requestwebanalysis + ", " + requestdatadownload + ", " + requestdataprocessing + ", " + requestprocessstartup + ", " + administer + ")", (OracleConnection)conn.Conn);
					cmd.ExecuteNonQuery();
				}
				catch (Exception)
				{
					return false;
				}
				return true;
			}
	
			/// <summary>
			/// Cleans orphan tokens.
			/// </summary>
			/// <param name="conn">the DB connection to be used.</param>
			/// <param name="trans">the DB transaction to be used.</param>
			/// <remarks>
			/// Releasing tokens is in the responsibility of OperaBatchManager. However, in specific conditions (typically after manual administrator interventions) it might happen that a process operation is closed and its token is not released. This function ensures that no orphan tokens are left.
			/// </remarks>
			public static void CleanOrphanTokens(OperaDbConnection conn, OperaDbTransaction trans)
			{
				new Oracle.ManagedDataAccess.Client.OracleCommand("CALL LP_CLEAN_ORPHAN_TOKENS()", (OracleConnection)conn.Conn).ExecuteNonQuery();
			}

			/// <summary>
			/// Ensures that a token has a specified owner.
			/// </summary>
			/// <param name="token">the token to be checked.</param>
			/// <param name="id_user">the Opera Id of the user whose ownership is to be verified. If this parameter is zero or negative, the username and password are used for verification.</param>
			/// <param name="username">the username of the user that claims ownership of the token. Ignored if <c>id_user</c> is positive.</param>
			/// <param name="password">the password of the user that claims ownership of the token. Ignored if <c>id_user</c> is positive.</param>
			/// <param name="conn">the DB connection to be used.</param>
			/// <param name="trans">the DB transaction to be used.</param>
			/// <remarks>If the verification fails, an exception is generated.</remarks>
			public static void CheckTokenOwnership(string token, long id_user, string username, string password, OperaDbConnection conn, OperaDbTransaction trans)
			{
				OracleCommand cmd = new Oracle.ManagedDataAccess.Client.OracleCommand("CALL LP_CHECK_TOKEN_OWNERSHIP('" + token + "', :id_user, :i_usr, :i_pwd)", (OracleConnection)conn.Conn);
				cmd.Parameters.Add("id_user", OracleDbType.Int64, ParameterDirection.InputOutput);
				if (id_user <= 0) cmd.Parameters[0].Value = System.DBNull.Value; else cmd.Parameters[0].Value = id_user;
				cmd.Parameters.Add("i_usr", OracleDbType.Varchar2, ParameterDirection.Input);
				if (username == null || username.Length == 0) cmd.Parameters[1].Value = System.DBNull.Value; else cmd.Parameters[1].Value = username;
				cmd.Parameters.Add("i_pwd", OracleDbType.Varchar2, ParameterDirection.Input);
				if (password == null || password.Length == 0) cmd.Parameters[2].Value = System.DBNull.Value; else cmd.Parameters[2].Value = password;
				cmd.ExecuteNonQuery();
			}

			/// <summary>
			/// Checks that a user has specific permissions.
			/// </summary>
			/// <param name="iduser">the DB identification number of the user.</param>
			/// <param name="rights">the list of the rights to be checked.</param>
			/// <param name="checkall">if true, the method checks that all permissions are granted; otherwise it checks that at least one of the permission is granted.</param>
			/// <param name="conn">the DB connection to be used.</param>
			/// <param name="trans">the DB transaction to be used. Can be null.</param>
			/// <returns>true if the user has the specified access, false otherwise.</returns>
			public static bool CheckAccess(long iduser, UserPermission [] rights, bool checkall, OperaDbConnection conn, OperaDbTransaction trans)
			{
				if (checkall)
				{
					OracleCommand cmd;
					foreach (UserPermission u in rights)
					{
						try
						{
							cmd = new OracleCommand("SELECT " + UserPermission.DBColumnName(u.Designator) + " FROM TB_PRIVILEGES WHERE (TB_PRIVILEGES.ID_USER=" + iduser + " AND TB_PRIVILEGES.ID_SITE=" + u.DB_Site_Id + ")", (OracleConnection)conn.Conn);
							if (Convert.ToInt64(cmd.ExecuteScalar()) == 0) return false;
						}
						catch (Exception)
						{
							return false;
						}
					}
					return true;
				}
				else
				{
					OracleCommand cmd;
					foreach (UserPermission u in rights)
					{
						try
						{
							cmd = new OracleCommand("SELECT " + UserPermission.DBColumnName(u.Designator) + " FROM TB_PRIVILEGES WHERE (TB_PRIVILEGES.ID_USER=" + iduser + " AND TB_PRIVILEGES.ID_SITE=" + u.DB_Site_Id + ")", (OracleConnection)conn.Conn);
							if (Convert.ToInt64(cmd.ExecuteScalar()) == 1) return true;
						}
						catch (Exception)
						{
							return false;
						}
					}
					return false;
				}
			}
			/// <summary>
			/// Protected constructor. Prevents users from creating instances of the User class without deriving it. Is implicitly called by constructors in derived classes.
			/// </summary>
			protected User() { Permissions = new UserPermission[0]; }
			/// <summary>
			/// Reads a user from an Opera DB.
			/// </summary>
			/// <param name="id">the DB identifier for the user entry to be read.</param>
			/// <param name="conn">the DB connection to be used.</param>
			/// <param name="trans">the DB transaction to be used. Can be null.</param>
			public User(long id, OperaDbConnection conn, OperaDbTransaction trans)
			{
				OracleDataAdapter da = new OracleDataAdapter("SELECT TB_USERS.USERNAME, TB_USERS.PWD, TB_USERS.NAME, TB_USERS.SURNAME, TB_USERS.INSTITUTION, TB_USERS.ID_SITE, TB_USERS.EMAIL, TB_USERS.ADDRESS, TB_USERS.PHONE FROM TB_USERS WHERE (TB_USERS.ID = " + id + ")", (OracleConnection)conn.Conn);
				DataSet ds = new DataSet();
				da.Fill(ds);
				DataRow dr = ds.Tables[0].Rows[0];
				m_DB_Id = id;
				m_UserName = dr[0].ToString();
				m_Password = dr[1].ToString();
				m_Name = dr[2].ToString();
				m_Surname = dr[3].ToString();
				m_Institution = dr[4].ToString();
				m_DB_Id_Site = Convert.ToInt64(dr[5]);
				m_EMailAddress = dr[6].ToString();
				m_Address = dr[7].ToString();
				m_PhoneNumber = dr[8].ToString();
				ds = new DataSet();
				System.Collections.ArrayList upl = new ArrayList();
				da.SelectCommand.CommandText = "SELECT ID_SITE, " + UserPermission.DBColumnName(UserPermissionDesignator.Scan) + ", " +					
					UserPermission.DBColumnName(UserPermissionDesignator.WebAnalysis) + ", " +
					UserPermission.DBColumnName(UserPermissionDesignator.ProcessData) + ", " +
					UserPermission.DBColumnName(UserPermissionDesignator.DownloadData) + ", " +
					UserPermission.DBColumnName(UserPermissionDesignator.StartupProcess) + ", " + 
					UserPermission.DBColumnName(UserPermissionDesignator.Administer) + " FROM TB_PRIVILEGES WHERE (TB_PRIVILEGES.ID_USER=" + id + ")";
				da.Fill(ds);
				foreach (DataRow drr in ds.Tables[0].Rows)
				{
					UserPermission up;
					up.DB_Site_Id = Convert.ToInt64(drr[0]);
					up.Designator = UserPermissionDesignator.Scan;
					up.Value = (Convert.ToInt16(drr[1]) == 1) ? UserPermissionTriState.Grant : UserPermissionTriState.Deny;
					upl.Add(up);
					up.Designator = UserPermissionDesignator.WebAnalysis;
					up.Value = (Convert.ToInt16(drr[2]) == 1) ? UserPermissionTriState.Grant : UserPermissionTriState.Deny;
					upl.Add(up);
					up.Designator = UserPermissionDesignator.ProcessData;
					up.Value = (Convert.ToInt16(drr[3]) == 1) ? UserPermissionTriState.Grant : UserPermissionTriState.Deny;
					upl.Add(up);
					up.Designator = UserPermissionDesignator.DownloadData;
					up.Value = (Convert.ToInt16(drr[4]) == 1) ? UserPermissionTriState.Grant : UserPermissionTriState.Deny;
					upl.Add(up);
					up.Designator = UserPermissionDesignator.StartupProcess;
					up.Value = (Convert.ToInt16(drr[5]) == 1) ? UserPermissionTriState.Grant : UserPermissionTriState.Deny;
					upl.Add(up);
					up.Designator = UserPermissionDesignator.Administer;
					up.Value = (Convert.ToInt16(drr[6]) == 1) ? UserPermissionTriState.Grant : UserPermissionTriState.Deny;
					upl.Add(up);
				}
				Permissions = (UserPermission [])upl.ToArray(typeof(UserPermission));
			}


			/// <summary>
			/// Reads a user from an Opera DB without accessing TB_USERS.
			/// This method needs the password and the site Id of the specific user.
			/// </summary>
			/// <param name="id">the DB identifier for the user entry to be read.</param>
			/// <param name="pwd">the Password for the user entry to be read.</param>
			/// <param name="site_id">the site identifier for the user entry to be read.</param>
			/// <param name="conn">the DB connection to be used.</param>
			/// <param name="trans">the DB transaction to be used. Can be null.</param>
			public User(long id,string pwd,long site_id,OperaDbConnection conn, OperaDbTransaction trans)
			{

				//Using VW_USERS
				OracleDataAdapter da = new OracleDataAdapter("SELECT VW_USERS.USERNAME, VW_USERS.NAME, VW_USERS.SURNAME, VW_USERS.INSTITUTION, VW_USERS.EMAIL, VW_USERS.ADDRESS, VW_USERS.PHONE FROM VW_USERS WHERE (VW_USERS.ID = " + id + ")", (OracleConnection)conn.Conn);
				DataSet ds = new DataSet();
				da.Fill(ds);
				DataRow dr = ds.Tables[0].Rows[0];
				m_DB_Id = id;
				m_UserName = dr[0].ToString();
				m_Password = pwd;
				m_Name = dr[1].ToString();
				m_Surname = dr[2].ToString();
				m_Institution = dr[3].ToString();
				m_DB_Id_Site = site_id;
				m_EMailAddress = dr[4].ToString();
				m_Address = dr[5].ToString();
				m_PhoneNumber = dr[6].ToString();
				
				//Using PC_GET_PRIVILEGES
				OracleCommand cmd= new OracleCommand("PC_GET_PRIVILEGES",(OracleConnection)conn.Conn);
				cmd.CommandType=CommandType.StoredProcedure;

				OracleParameter prm1=new OracleParameter("u_userid",OracleDbType.Int64);
				prm1.Direction=ParameterDirection.Input;
				prm1.Value=id;
				cmd.Parameters.Add(prm1);

				OracleParameter prm2=new OracleParameter("s_siteid",OracleDbType.Int64);
				prm2.Direction=ParameterDirection.Input;
				prm2.Value=site_id;
				cmd.Parameters.Add(prm2);

				OracleParameter prm3=new OracleParameter("pwd",OracleDbType.Varchar2);
				prm3.Direction=ParameterDirection.Input;
				prm3.Value=pwd;	
				cmd.Parameters.Add(prm3);

				OracleParameter prm4=new OracleParameter("p_scan",OracleDbType.Int16);
				prm4.Direction=ParameterDirection.Output;
				cmd.Parameters.Add(prm4);
								
				OracleParameter prm5=new OracleParameter("p_weban",OracleDbType.Int16);
				prm5.Direction=ParameterDirection.Output;
				cmd.Parameters.Add(prm5);

				OracleParameter prm6=new OracleParameter("p_dataproc",OracleDbType.Int16);
				prm6.Direction=ParameterDirection.Output;
				cmd.Parameters.Add(prm6);
								
				OracleParameter prm7=new OracleParameter("p_datadwnl",OracleDbType.Int16);
				prm7.Direction=ParameterDirection.Output;
				cmd.Parameters.Add(prm7);
				
				OracleParameter prm8=new OracleParameter("p_procstart",OracleDbType.Int16);
				prm8.Direction=ParameterDirection.Output;
				cmd.Parameters.Add(prm8);

				OracleParameter prm9=new OracleParameter("p_admin",OracleDbType.Int16);
				prm9.Direction=ParameterDirection.Output;
				cmd.Parameters.Add(prm9);

				cmd.ExecuteNonQuery();

				//Getting user privileges
				System.Collections.ArrayList upl = new ArrayList();

				UserPermission up;
				
				up.DB_Site_Id = site_id;
				up.Designator = UserPermissionDesignator.Scan;
				up.Value = (Convert.ToInt16(cmd.Parameters["p_scan"].Value) == 1) ? UserPermissionTriState.Grant : UserPermissionTriState.Deny;
				upl.Add(up);

				up.Designator = UserPermissionDesignator.WebAnalysis;
				up.Value = (Convert.ToInt16(cmd.Parameters["p_weban"].Value) == 1) ? UserPermissionTriState.Grant : UserPermissionTriState.Deny;
				upl.Add(up);
				
				up.Designator = UserPermissionDesignator.ProcessData;
				up.Value = (Convert.ToInt16(cmd.Parameters["p_dataproc"].Value) == 1) ? UserPermissionTriState.Grant : UserPermissionTriState.Deny;
				upl.Add(up);
				
				up.Designator = UserPermissionDesignator.DownloadData;
				up.Value = (Convert.ToInt16(cmd.Parameters["p_datadwnl"].Value) == 1) ? UserPermissionTriState.Grant : UserPermissionTriState.Deny;
				upl.Add(up);
				
				up.Designator = UserPermissionDesignator.StartupProcess;
				up.Value = (Convert.ToInt16(cmd.Parameters["p_procstart"].Value) == 1) ? UserPermissionTriState.Grant : UserPermissionTriState.Deny;
				upl.Add(up);
				
				up.Designator = UserPermissionDesignator.Administer;
				up.Value = (Convert.ToInt16(cmd.Parameters["p_admin"].Value) == 1) ? UserPermissionTriState.Grant : UserPermissionTriState.Deny;
				upl.Add(up);
				
				Permissions = (UserPermission [])upl.ToArray(typeof(UserPermission));
						
			}

			/// <summary>
			/// Reads a user from an Opera DB without accessig TB_USERS.
			/// This method needs the password and the site Id of the specific user.
			/// </summary>
			/// <param name="adm_id">the DB identifier for the administrator.</param>
			/// <param name="adm_pwd">the Password for the administrator.</param>
			/// <param name="u_id">the DB identifier for the user entry to be read.</param>
			/// <param name="site_id">the site identifier for the user entry to be read.</param>
			/// <param name="conn">the DB connection to be used.</param>
			/// <param name="trans">the DB transaction to be used. Can be null.</param>
			public User(long adm_id, string adm_pwd,long u_id,long site_id,OperaDbConnection conn, OperaDbTransaction trans)
			{

				//Getting user pwd using PC_GET_PWD

				string u_pwd="";

				OracleCommand cmd= new OracleCommand("PC_GET_PWD",(OracleConnection)conn.Conn);
				cmd.CommandType=CommandType.StoredProcedure;

				OracleParameter prm01=new OracleParameter("admin_id",OracleDbType.Int64);
				prm01.Direction=ParameterDirection.Input;
				prm01.Value=adm_id;
				cmd.Parameters.Add(prm01);

				OracleParameter prm02=new OracleParameter("admin_pwd",OracleDbType.Varchar2);
				prm02.Direction=ParameterDirection.Input;
				prm02.Value=adm_pwd;
				cmd.Parameters.Add(prm02);

				OracleParameter prm03=new OracleParameter("user_id",OracleDbType.Int64);
				prm03.Direction=ParameterDirection.Input;
				prm03.Value=u_id;
				cmd.Parameters.Add(prm03);

				OracleParameter prm04=new OracleParameter("user_pwd",OracleDbType.Varchar2,50);
				prm04.Direction=ParameterDirection.Output;
				cmd.Parameters.Add(prm04);

				cmd.ExecuteNonQuery();

				u_pwd=cmd.Parameters["user_pwd"].Value.ToString();
				
				//Using VW_USERS
				OracleDataAdapter da = new OracleDataAdapter("SELECT VW_USERS.USERNAME, VW_USERS.NAME, VW_USERS.SURNAME, VW_USERS.INSTITUTION, VW_USERS.EMAIL, VW_USERS.ADDRESS, VW_USERS.PHONE FROM VW_USERS WHERE (VW_USERS.ID = " + u_id + ")", (OracleConnection)conn.Conn);
				DataSet ds = new DataSet();
				da.Fill(ds);
				DataRow dr = ds.Tables[0].Rows[0];
				m_DB_Id = u_id;
				m_UserName = dr[0].ToString();
				m_Password = u_pwd;
				m_Name = dr[1].ToString();
				m_Surname = dr[2].ToString();
				m_Institution = dr[3].ToString();
				m_DB_Id_Site = site_id;
				m_EMailAddress = dr[4].ToString();
				m_Address = dr[5].ToString();
				m_PhoneNumber = dr[6].ToString();
				
				//Using PC_GET_PRIVILEGES_ADM
				cmd= new OracleCommand("PC_GET_PRIVILEGES_ADM",(OracleConnection)conn.Conn);
				cmd.CommandType=CommandType.StoredProcedure;

				OracleParameter prm1a=new OracleParameter("adm_id",OracleDbType.Int64);
				prm1a.Direction=ParameterDirection.Input;
				prm1a.Value=adm_id;
				cmd.Parameters.Add(prm1a);

				OracleParameter prm1b=new OracleParameter("adm_pwd",OracleDbType.Varchar2);
				prm1b.Direction=ParameterDirection.Input;
				prm1b.Value=adm_pwd;
				cmd.Parameters.Add(prm1b);

				OracleParameter prm2=new OracleParameter("u_userid",OracleDbType.Int64);
				prm2.Direction=ParameterDirection.Input;
				prm2.Value=u_id;
				cmd.Parameters.Add(prm2);

				OracleParameter prm3=new OracleParameter("s_siteid",OracleDbType.Int64);
				prm3.Direction=ParameterDirection.Input;
				prm3.Value=site_id;
				cmd.Parameters.Add(prm3);

				OracleParameter prm4=new OracleParameter("p_scan",OracleDbType.Int16);
				prm4.Direction=ParameterDirection.Output;
				cmd.Parameters.Add(prm4);
								
				OracleParameter prm5=new OracleParameter("p_weban",OracleDbType.Int16);
				prm5.Direction=ParameterDirection.Output;
				cmd.Parameters.Add(prm5);

				OracleParameter prm6=new OracleParameter("p_dataproc",OracleDbType.Int16);
				prm6.Direction=ParameterDirection.Output;
				cmd.Parameters.Add(prm6);
								
				OracleParameter prm7=new OracleParameter("p_datadwnl",OracleDbType.Int16);
				prm7.Direction=ParameterDirection.Output;
				cmd.Parameters.Add(prm7);
				
				OracleParameter prm8=new OracleParameter("p_procstart",OracleDbType.Int16);
				prm8.Direction=ParameterDirection.Output;
				cmd.Parameters.Add(prm8);

				OracleParameter prm9=new OracleParameter("p_admin",OracleDbType.Int16);
				prm9.Direction=ParameterDirection.Output;
				cmd.Parameters.Add(prm9);

				//executing query
				cmd.ExecuteNonQuery();

				//getting user privileges
				System.Collections.ArrayList upl = new ArrayList();

				UserPermission up;
				
				up.DB_Site_Id = site_id;
				up.Designator = UserPermissionDesignator.Scan;
				up.Value = (Convert.ToInt16(cmd.Parameters["p_scan"].Value) == 1) ? UserPermissionTriState.Grant : UserPermissionTriState.Deny;
				upl.Add(up);

				up.Designator = UserPermissionDesignator.WebAnalysis;
				up.Value = (Convert.ToInt16(cmd.Parameters["p_weban"].Value) == 1) ? UserPermissionTriState.Grant : UserPermissionTriState.Deny;
				upl.Add(up);
				
				up.Designator = UserPermissionDesignator.ProcessData;
				up.Value = (Convert.ToInt16(cmd.Parameters["p_dataproc"].Value) == 1) ? UserPermissionTriState.Grant : UserPermissionTriState.Deny;
				upl.Add(up);
				
				up.Designator = UserPermissionDesignator.DownloadData;
				up.Value = (Convert.ToInt16(cmd.Parameters["p_datadwnl"].Value) == 1) ? UserPermissionTriState.Grant : UserPermissionTriState.Deny;
				upl.Add(up);
				
				up.Designator = UserPermissionDesignator.StartupProcess;
				up.Value = (Convert.ToInt16(cmd.Parameters["p_procstart"].Value) == 1) ? UserPermissionTriState.Grant : UserPermissionTriState.Deny;
				upl.Add(up);
				
				up.Designator = UserPermissionDesignator.Administer;
				up.Value = (Convert.ToInt16(cmd.Parameters["p_admin"].Value) == 1) ? UserPermissionTriState.Grant : UserPermissionTriState.Deny;
				upl.Add(up);
				
				Permissions = (UserPermission [])upl.ToArray(typeof(UserPermission));
						
			}


			/// <summary>
			/// Adds a user entry to an Opera DB.
			/// </summary>
			/// <param name="conn">the DB connection to be used.</param>
			/// <param name="trans">the DB transaction to be used.</param>
			/// <returns>the DB identifier that has been assigned to the user.</returns>
			public long Add(OperaDbConnection conn, OperaDbTransaction trans)
			{
				//using PC_ADD_USER

				OracleCommand cmd= new OracleCommand("PC_ADD_USER",(OracleConnection)conn.Conn);
				cmd.CommandType=CommandType.StoredProcedure;

				//setting parameters
				OracleParameter prm1=new OracleParameter("s_id",OracleDbType.Int64);
				prm1.Direction=ParameterDirection.Input;
				prm1.Value=DB_Id_Site;
				cmd.Parameters.Add(prm1);

				OracleParameter prm2=new OracleParameter("u_username",OracleDbType.Varchar2);
				prm2.Direction=ParameterDirection.Input;
				prm2.Value=UserName.ToLower();
				cmd.Parameters.Add(prm2);

				OracleParameter prm3=new OracleParameter("u_pwd",OracleDbType.Varchar2);
				prm3.Direction=ParameterDirection.Input;
				prm3.Value=Password;	
				cmd.Parameters.Add(prm3);

				OracleParameter prm4=new OracleParameter("u_name",OracleDbType.Varchar2);
				prm4.Direction=ParameterDirection.Input;
				prm4.Value=Name;
				cmd.Parameters.Add(prm4);
								
				OracleParameter prm5=new OracleParameter("u_surname",OracleDbType.Varchar2);
				prm5.Direction=ParameterDirection.Input;
				prm5.Value=Surname;
				cmd.Parameters.Add(prm5);

				OracleParameter prm6=new OracleParameter("u_inst",OracleDbType.Varchar2);
				prm6.Direction=ParameterDirection.Input;
				prm6.Value=Institution;
				cmd.Parameters.Add(prm6);
								
				OracleParameter prm7=new OracleParameter("u_email",OracleDbType.Varchar2);
				prm7.Direction=ParameterDirection.Input;
				prm7.Value=EMailAddress;
				cmd.Parameters.Add(prm7);
				
				OracleParameter prm8=new OracleParameter("u_address",OracleDbType.Varchar2);
				prm8.Direction=ParameterDirection.Input;
				prm8.Value=Address;
				cmd.Parameters.Add(prm8);

				OracleParameter prm9=new OracleParameter("u_phone",OracleDbType.Varchar2);
				prm9.Direction=ParameterDirection.Input;
				prm9.Value=PhoneNumber;
				cmd.Parameters.Add(prm9);

				OracleParameter prm10=new OracleParameter("newid",OracleDbType.Int64);
				prm10.Direction=ParameterDirection.Output;
				cmd.Parameters.Add(prm10);

				//executing query
				cmd.ExecuteNonQuery();

				m_DB_Id = Convert.ToInt64(cmd.Parameters["newid"].Value);

				//PC_SET_PRIVILEGES : all permissions set to Deny
				cmd= new OracleCommand("PC_SET_PRIVILEGES",(OracleConnection)conn.Conn);
				cmd.CommandType=CommandType.StoredProcedure;

				//setting parameters
				OracleParameter prm11=new OracleParameter("u_id",OracleDbType.Int64);
				prm11.Direction=ParameterDirection.Input;
				prm11.Value=DB_Id;
				cmd.Parameters.Add(prm11);

				OracleParameter prm12=new OracleParameter("s_id",OracleDbType.Int64);
				prm12.Direction=ParameterDirection.Input;
				prm12.Value=DB_Id_Site;
				cmd.Parameters.Add(prm12);

				OracleParameter prm13=new OracleParameter("p_scan",OracleDbType.Int16);
				prm13.Direction=ParameterDirection.Input;
				prm13.Value=0;	
				cmd.Parameters.Add(prm13);

				OracleParameter prm14=new OracleParameter("p_weban",OracleDbType.Int16);
				prm14.Direction=ParameterDirection.Input;
				prm14.Value=0;
				cmd.Parameters.Add(prm14);
								
				OracleParameter prm15=new OracleParameter("p_dataproc",OracleDbType.Int16);
				prm15.Direction=ParameterDirection.Input;
				prm15.Value=0;
				cmd.Parameters.Add(prm15);

				OracleParameter prm16=new OracleParameter("p_datadwnl",OracleDbType.Int16);
				prm16.Direction=ParameterDirection.Input;
				prm16.Value=0;
				cmd.Parameters.Add(prm16);
								
				OracleParameter prm17=new OracleParameter("p_procstart",OracleDbType.Int16);
				prm17.Direction=ParameterDirection.Input;
				prm17.Value=0;
				cmd.Parameters.Add(prm17);
				
				OracleParameter prm18=new OracleParameter("p_admin",OracleDbType.Int16);
				prm18.Direction=ParameterDirection.Input;
				prm18.Value=0;
				cmd.Parameters.Add(prm18);


				//set permissions if present
				foreach (UserPermission upp in Permissions)
				{
					if (upp.Designator==UserPermissionDesignator.Scan)
					{
						prm13.Value=(upp.Value == UserPermissionTriState.Grant) ? 1 : 0;	
					}
					if (upp.Designator==UserPermissionDesignator.WebAnalysis)
					{
						prm14.Value=(upp.Value == UserPermissionTriState.Grant) ? 1 : 0;	
					}
					if (upp.Designator==UserPermissionDesignator.ProcessData)
					{
						prm15.Value=(upp.Value == UserPermissionTriState.Grant) ? 1 : 0;	
					}
					if (upp.Designator==UserPermissionDesignator.DownloadData)
					{
						prm16.Value=(upp.Value == UserPermissionTriState.Grant) ? 1 : 0;	
					}
					if (upp.Designator==UserPermissionDesignator.StartupProcess)
					{
						prm17.Value=(upp.Value == UserPermissionTriState.Grant) ? 1 : 0;	
					}
					if (upp.Designator==UserPermissionDesignator.Administer)
					{
						prm18.Value=(upp.Value == UserPermissionTriState.Grant) ? 1 : 0;	
					}
				}

				cmd.ExecuteNonQuery();

				return m_DB_Id;			
			}
			/// <summary>
			/// Synchronizes a modified user with its DB entry.
			/// The DB_Id must have been left unmodified.
			/// </summary>
			/// <param name="conn">the DB connection to be used.</param>
			/// <param name="trans">the DB transaction to be used. Cannot be null.</param>
			public void Synchronize(OperaDbConnection conn, OperaDbTransaction trans)
			{

				//using PC_SET_USER

				OracleCommand cmd= new OracleCommand("PC_SET_USER",(OracleConnection)conn.Conn);
				cmd.CommandType=CommandType.StoredProcedure;

				//setting parameters
				OracleParameter prm1=new OracleParameter("u_id",OracleDbType.Int64);
				prm1.Direction=ParameterDirection.Input;
				prm1.Value=DB_Id;
				cmd.Parameters.Add(prm1);

				OracleParameter prm2=new OracleParameter("u_username",OracleDbType.Varchar2);
				prm2.Direction=ParameterDirection.Input;
				prm2.Value=UserName.ToLower();
				cmd.Parameters.Add(prm2);

				OracleParameter prm3=new OracleParameter("u_pwd",OracleDbType.Varchar2);
				prm3.Direction=ParameterDirection.Input;
				prm3.Value=Password;	
				cmd.Parameters.Add(prm3);

				OracleParameter prm4=new OracleParameter("u_name",OracleDbType.Varchar2);
				prm4.Direction=ParameterDirection.Input;
				prm4.Value=Name;
				cmd.Parameters.Add(prm4);
								
				OracleParameter prm5=new OracleParameter("u_surname",OracleDbType.Varchar2);
				prm5.Direction=ParameterDirection.Input;
				prm5.Value=Surname;
				cmd.Parameters.Add(prm5);

				OracleParameter prm6=new OracleParameter("u_inst",OracleDbType.Varchar2);
				prm6.Direction=ParameterDirection.Input;
				prm6.Value=Institution;
				cmd.Parameters.Add(prm6);
								
				OracleParameter prm7=new OracleParameter("u_email",OracleDbType.Varchar2);
				prm7.Direction=ParameterDirection.Input;
				prm7.Value=EMailAddress;
				cmd.Parameters.Add(prm7);
				
				OracleParameter prm8=new OracleParameter("u_address",OracleDbType.Varchar2);
				prm8.Direction=ParameterDirection.Input;
				prm8.Value=Address;
				cmd.Parameters.Add(prm8);

				OracleParameter prm9=new OracleParameter("u_phone",OracleDbType.Varchar2);
				prm9.Direction=ParameterDirection.Input;
				prm9.Value=PhoneNumber;
				cmd.Parameters.Add(prm9);

				//executing query
				cmd.ExecuteNonQuery();

				//using PC_DEL_PRIVILEGES

				// using oracle stored procedure
				cmd= new OracleCommand("PC_DEL_PRIVILEGES",(OracleConnection)conn.Conn);
				cmd.CommandType=CommandType.StoredProcedure;

				//setting parameters
				OracleParameter prm11=new OracleParameter("u_id",OracleDbType.Int64);
				prm11.Direction=ParameterDirection.Input;
				prm11.Value=DB_Id;
				cmd.Parameters.Add(prm11);

				OracleParameter prm12=new OracleParameter("s_id",OracleDbType.Int64);
				prm12.Direction=ParameterDirection.Input;
				prm12.Value=DB_Id_Site;
				cmd.Parameters.Add(prm12);

				cmd.ExecuteNonQuery();

				//Using PC_SET_PRVILEGES

				//PC_SET_PRIVILEGES : all permissions to Deny
				cmd= new OracleCommand("PC_SET_PRIVILEGES",(OracleConnection)conn.Conn);
				cmd.CommandType=CommandType.StoredProcedure;

				//setting parameters
				OracleParameter prm21=new OracleParameter("u_id",OracleDbType.Int64);
				prm21.Direction=ParameterDirection.Input;
				prm21.Value=DB_Id;
				cmd.Parameters.Add(prm21);

				OracleParameter prm22=new OracleParameter("s_id",OracleDbType.Int64);
				prm22.Direction=ParameterDirection.Input;
				prm22.Value=DB_Id_Site;
				cmd.Parameters.Add(prm22);

				OracleParameter prm23=new OracleParameter("p_scan",OracleDbType.Int16);
				prm23.Direction=ParameterDirection.Input;
				prm23.Value=0;	
				cmd.Parameters.Add(prm23);

				OracleParameter prm24=new OracleParameter("p_weban",OracleDbType.Int16);
				prm24.Direction=ParameterDirection.Input;
				prm24.Value=0;
				cmd.Parameters.Add(prm24);
								
				OracleParameter prm25=new OracleParameter("p_dataproc",OracleDbType.Int16);
				prm25.Direction=ParameterDirection.Input;
				prm25.Value=0;
				cmd.Parameters.Add(prm25);

				OracleParameter prm26=new OracleParameter("p_datadwnl",OracleDbType.Int16);
				prm26.Direction=ParameterDirection.Input;
				prm26.Value=0;
				cmd.Parameters.Add(prm26);
								
				OracleParameter prm27=new OracleParameter("p_procstart",OracleDbType.Int16);
				prm27.Direction=ParameterDirection.Input;
				prm27.Value=0;
				cmd.Parameters.Add(prm27);
				
				OracleParameter prm28=new OracleParameter("p_admin",OracleDbType.Int16);
				prm28.Direction=ParameterDirection.Input;
				prm28.Value=0;
				cmd.Parameters.Add(prm28);


				//Setting permissions if present
				foreach (UserPermission upp in Permissions)
				{
					if (upp.Designator==UserPermissionDesignator.Scan)
					{
						prm23.Value=(upp.Value == UserPermissionTriState.Grant) ? 1 : 0;	
					}
					if (upp.Designator==UserPermissionDesignator.WebAnalysis)
					{
						prm24.Value=(upp.Value == UserPermissionTriState.Grant) ? 1 : 0;	
					}
					if (upp.Designator==UserPermissionDesignator.ProcessData)
					{
						prm25.Value=(upp.Value == UserPermissionTriState.Grant) ? 1 : 0;	
					}
					if (upp.Designator==UserPermissionDesignator.DownloadData)
					{
						prm26.Value=(upp.Value == UserPermissionTriState.Grant) ? 1 : 0;	
					}
					if (upp.Designator==UserPermissionDesignator.StartupProcess)
					{
						prm27.Value=(upp.Value == UserPermissionTriState.Grant) ? 1 : 0;	
					}
					if (upp.Designator==UserPermissionDesignator.Administer)
					{
						prm28.Value=(upp.Value == UserPermissionTriState.Grant) ? 1 : 0;	
					}
				}

				cmd.ExecuteNonQuery();

			}

			/// <summary>
			/// Deletes a user from the Opera DB using oracle stored procedure.
			/// </summary>
			/// <param name="id">the DB identifier of the user to be deleted.</param>
			/// <param name="siteId">the DB site identifier of the user to be deleted.</param>
			/// <param name="conn">the DB connection to be used.</param>
			/// <param name="trans">the DB transaction to be used. Cannot be null.</param>
			public static void Delete(long id,long siteId ,OperaDbConnection conn, OperaDbTransaction trans)
			{
				OperaDbTransaction mytrans = (trans == null) ? conn.BeginTransaction() : trans;
				try
				{
				
					//Using PC_DEL_PRIVILEGES
					OracleCommand cmd= new OracleCommand("PC_DEL_PRIVILEGES",(OracleConnection)conn.Conn);
					cmd.CommandType=CommandType.StoredProcedure;

					OracleParameter prm1=new OracleParameter("u_id",OracleDbType.Int64);
					prm1.Direction=ParameterDirection.Input;
					prm1.Value=id;
					cmd.Parameters.Add(prm1);

					OracleParameter prm2=new OracleParameter("s_id",OracleDbType.Int64);
					prm2.Direction=ParameterDirection.Input;
					prm2.Value=siteId;
					cmd.Parameters.Add(prm2);

					cmd.ExecuteNonQuery();

					//Using PC_DEL_USER
					cmd= new OracleCommand("PC_DEL_USER",(OracleConnection)conn.Conn);
					cmd.CommandType=CommandType.StoredProcedure;

					OracleParameter prm11=new OracleParameter("u_id",OracleDbType.Int64);
					prm11.Direction=ParameterDirection.Input;
					prm11.Value=id;
					cmd.Parameters.Add(prm11);

					cmd.ExecuteNonQuery();

					if (trans == null) mytrans.Commit();
				}
				catch (Exception x)
				{
					if (trans == null) mytrans.Rollback();
					throw x;
				}
			}


			/// <summary>
			/// Deletes a user from the Opera DB.
			/// </summary>
			/// <param name="id">the DB identifier of the user to be deleted.</param>
			/// <param name="conn">the DB connection to be used.</param>
			/// <param name="trans">the DB transaction to be used. Cannot be null.</param>
			public static void Delete(long id, OperaDbConnection conn, OperaDbTransaction trans)
			{
				OperaDbTransaction mytrans = (trans == null) ? conn.BeginTransaction() : trans;
				try
				{
					new OracleCommand("DELETE FROM TB_PRIVILEGES WHERE (TB_PRIVILEGES.ID_USER = " + id + ")", (OracleConnection)conn.Conn).ExecuteNonQuery();
					new OracleCommand("DELETE FROM TB_USERS WHERE (TB_USERS.ID = " + id + ")", (OracleConnection)conn.Conn).ExecuteNonQuery();
					if (trans == null) mytrans.Commit();
				}
				catch (Exception x)
				{
					if (trans == null) mytrans.Rollback();
					throw x;
				}
			}
		}

		/// <summary>
		/// A complex of program settings for scanning or data processing in an OperaDb.
		/// </summary>
		public class ProgramSettings
		{
			/// <summary>
			/// Protected data member on which the DB_Id property relies. Can be accessed from derived classes.
			/// </summary>
			protected long m_DB_Id;
			/// <summary>
			/// Opera DB identifier of the program settings entry.
			/// </summary>
			public long DB_Id { get { return m_DB_Id; } }
			/// <summary>
			/// Protected data member on which the DB_Id_Author property relies. Can be accessed from derived classes.
			/// </summary>
			protected long m_DB_Id_Author;
			/// <summary>
			/// Opera DB identifier of the author that produced the settings.
			/// </summary>
			public long DB_Id_Author { get { return m_DB_Id_Author; } }
			/// <summary>
			/// Protected data member on which the Description property relies. Can be accessed from derived classes.
			/// </summary>
			protected string m_Description;
			/// <summary>
			/// Description of the program settings.
			/// </summary>
			public string Description { get { return m_Description; } }
			/// <summary>
			/// Protected data member on which the Executable property relies. Can be accessed from derived classes.
			/// </summary>
			protected string m_Executable;
			/// <summary>
			/// Executable of the program settings.
			/// </summary>
			public string Executable { get { return m_Executable; } }
			/// <summary>
			/// The member on which DriverLevel relies. Can be accessed by derived classes.
			/// </summary>
			protected SySal.DAQSystem.Drivers.DriverType m_DriverLevel;
			/// <summary>
			/// Level of the driver to be executed.
			/// </summary>
			public SySal.DAQSystem.Drivers.DriverType DriverLevel { get { return m_DriverLevel; } }
			/// <summary>
			/// The member on which UsesTemplateMarks relies. Can be accessed by derived classes.
			/// </summary>
			protected bool m_UsesTemplateMarks;
			/// <summary>
			/// Tells whether this operation uses template marks. It is meaningful only for Scanning drivers.
			/// </summary>
			public bool UsesTemplateMarks { get { return m_UsesTemplateMarks; } }
			/// <summary>
			/// Protected data member on which the ModuleName property relies. Can be accessed from derived classes.
			/// </summary>
			protected string m_Settings;
			/// <summary>
			/// Data that specify executable assemblies and configuration parameters.
			/// </summary>
			public string Settings { get { return m_Settings; } }
			/// <summary>
			/// Protected constructor. Prevents users from creating instances of the ProgramSettings class without deriving it. Is implicitly called by constructors in derived classes.
			/// </summary>
			protected ProgramSettings() {}
			/// <summary>
			/// Reads program settings from an Opera DB.
			/// </summary>
			/// <param name="id">the DB identifier for the program settings entry to be read.</param>
			/// <param name="conn">the DB connection to be used.</param>
			/// <param name="trans">the DB transaction to be used. Can be null.</param>
			public ProgramSettings(long id, OperaDbConnection conn, OperaDbTransaction trans)
			{
				DataSet ds = new DataSet();
				OracleDataAdapter da = new OracleDataAdapter("SELECT TB_PROGRAMSETTINGS.DESCRIPTION, TB_PROGRAMSETTINGS.EXECUTABLE, TB_PROGRAMSETTINGS.ID_AUTHOR, TB_PROGRAMSETTINGS.DRIVERLEVEL, TB_PROGRAMSETTINGS.TEMPLATEMARKS, TB_PROGRAMSETTINGS.SETTINGS FROM TB_PROGRAMSETTINGS WHERE (ID = " + id.ToString() + ")", (OracleConnection)conn.Conn);
				da.Fill(ds);
				m_DB_Id = id;
				m_Description = ds.Tables[0].Rows[0][0].ToString();
				m_Executable = ds.Tables[0].Rows[0][1].ToString();
				m_DB_Id_Author = Convert.ToInt64(ds.Tables[0].Rows[0][2]);
				m_DriverLevel = (SySal.DAQSystem.Drivers.DriverType)Convert.ToInt64(ds.Tables[0].Rows[0][3]);
				m_UsesTemplateMarks = (ds.Tables[0].Rows[0][4] == System.DBNull.Value) ? true : (Convert.ToInt64(ds.Tables[0].Rows[0][4]) != 0);
				m_Settings = ds.Tables[0].Rows[0][5].ToString();
			}

			/// <summary>
			/// Adds a program settings entry to an Opera DB.
			/// </summary>
			/// <param name="conn">the DB connection to be used.</param>
			/// <param name="trans">the DB transaction to be used.</param>
			/// <returns>the DB identifier that has been assigned to the program settings.</returns>
			public long Add(OperaDbConnection conn, OperaDbTransaction trans)
			{
				OracleCommand cmd = new OracleCommand("CALL PC_ADD_PROGRAMSETTINGS(:descr, :exe, :idauth, :drvlev, :templmks, :settings, :newid)", (OracleConnection)conn.Conn);
				cmd.Parameters.Add("descr", Oracle.ManagedDataAccess.Client.OracleDbType.Varchar2, System.Data.ParameterDirection.Input).Value = m_Description;
				cmd.Parameters.Add("exe", Oracle.ManagedDataAccess.Client.OracleDbType.Varchar2, System.Data.ParameterDirection.Input).Value = m_Executable;
				cmd.Parameters.Add("idauth", Oracle.ManagedDataAccess.Client.OracleDbType.Int64, System.Data.ParameterDirection.Input).Value = m_DB_Id_Author;
				cmd.Parameters.Add("drvlev", Oracle.ManagedDataAccess.Client.OracleDbType.Int64, System.Data.ParameterDirection.Input).Value = m_DriverLevel;
				cmd.Parameters.Add("templmks", Oracle.ManagedDataAccess.Client.OracleDbType.Int64, System.Data.ParameterDirection.Input);
				if (m_DriverLevel == SySal.DAQSystem.Drivers.DriverType.Scanning) cmd.Parameters[4].Value = m_UsesTemplateMarks ? 1 : 0;
				else cmd.Parameters[4].Value =  System.DBNull.Value;
				cmd.Parameters.Add("settings", Oracle.ManagedDataAccess.Client.OracleDbType.Clob, System.Data.ParameterDirection.Input).Value = m_Settings;
				cmd.Parameters.Add("newid", Oracle.ManagedDataAccess.Client.OracleDbType.Int64, System.Data.ParameterDirection.Output);
				cmd.ExecuteNonQuery();
				return m_DB_Id = Convert.ToInt64(cmd.Parameters[6].Value);
			}
		}
	}
}
