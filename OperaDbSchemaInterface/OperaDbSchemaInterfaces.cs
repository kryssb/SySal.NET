namespace SySal.OperaDb
{
	/// <summary>
	/// Static class that allows access to schema components of the OPERA DB.
	/// Its DB field must be set before the class can be used.
	/// </summary>
	public class Schema
	{
		private Schema() {}
		/// <summary>
		/// Retrieval order for query results.
		/// </summary>
		public enum OrderBy { None, Ascending, Descending }
		/// <summary>
		/// Accesses the LZ_BUFFERFLUSH table in the DB.
		/// For data insertion, the Insert and Flush static methods are used. This class inserts data in batches.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class LZ_BUFFERFLUSH
		{
			internal LZ_BUFFERFLUSH() {}
			System.Data.DataRowCollection m_DRC;
			const int ArraySize = 100;
			static int index = 0;
			/// <summary>
			/// Since this class uses batch insertion, this method must be called to ensure that all rows are actually inserted into LZ_BUFFERFLUSH. Failure to do so will result in incomplete writes.
			/// </summary>
			static public void Flush() { if (cmd == null || index == 0) return; cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			private static object [] a_STATUS = new object[ArraySize];
			/// <summary>
			/// Retrieves STATUS for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _STATUS
			{
				get
				{
					if (m_DR[0] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is actually inserted later, in a batch insertion command. The Flush method must be called to ensure all rows are actually written.
			/// </summary>
			/// <param name="i_STATUS">the value to be inserted for STATUS. The value for this parameter can be long or System.DBNull.Value.</param>
			static public void Insert(object i_STATUS)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				a_STATUS[index] = i_STATUS;
				if (++index >= ArraySize) { cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			}
			/// <summary>
			/// Reads a set of rows from LZ_BUFFERFLUSH and retrieves them into a new LZ_BUFFERFLUSH object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the LZ_BUFFERFLUSH class that can be used to read the retrieved data.</returns>
			static public LZ_BUFFERFLUSH SelectWhere(string wherestr, string orderstr)
			{
				LZ_BUFFERFLUSH newobj = new LZ_BUFFERFLUSH();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT STATUS FROM LZ_BUFFERFLUSH" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO LZ_BUFFERFLUSH (STATUS) VALUES (:p_1)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_STATUS;
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the LZ_GRAINS table in the DB.
		/// For data insertion, the Insert and Flush static methods are used. This class inserts data in batches.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class LZ_GRAINS
		{
			internal LZ_GRAINS() {}
			System.Data.DataRowCollection m_DRC;
			const int ArraySize = 100;
			static int index = 0;
			/// <summary>
			/// Since this class uses batch insertion, this method must be called to ensure that all rows are actually inserted into LZ_GRAINS. Failure to do so will result in incomplete writes.
			/// </summary>
			static public void Flush() { if (cmd == null || index == 0) return; cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			private static object [] a_ID_EVENTBRICK = new object[ArraySize];
			/// <summary>
			/// Retrieves ID_EVENTBRICK for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _ID_EVENTBRICK
			{
				get
				{
					if (m_DR[0] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			private static object [] a_ID_ZONE = new object[ArraySize];
			/// <summary>
			/// Retrieves ID_ZONE for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _ID_ZONE
			{
				get
				{
					if (m_DR[1] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[1]);
				}
			}
			private static object [] a_SIDE = new object[ArraySize];
			/// <summary>
			/// Retrieves SIDE for the current row. The return value can be System.DBNull.Value or a value that can be cast to int.
			/// </summary>
			public object _SIDE
			{
				get
				{
					if (m_DR[2] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt32(m_DR[2]);
				}
			}
			private static object [] a_ID_MIPMICROTRACK = new object[ArraySize];
			/// <summary>
			/// Retrieves ID_MIPMICROTRACK for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _ID_MIPMICROTRACK
			{
				get
				{
					if (m_DR[3] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[3]);
				}
			}
			private static object [] a_ID = new object[ArraySize];
			/// <summary>
			/// Retrieves ID for the current row. The return value can be System.DBNull.Value or a value that can be cast to double.
			/// </summary>
			public object _ID
			{
				get
				{
					if (m_DR[4] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDouble(m_DR[4]);
				}
			}
			private static object [] a_X = new object[ArraySize];
			/// <summary>
			/// Retrieves X for the current row. The return value can be System.DBNull.Value or a value that can be cast to double.
			/// </summary>
			public object _X
			{
				get
				{
					if (m_DR[5] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDouble(m_DR[5]);
				}
			}
			private static object [] a_Y = new object[ArraySize];
			/// <summary>
			/// Retrieves Y for the current row. The return value can be System.DBNull.Value or a value that can be cast to double.
			/// </summary>
			public object _Y
			{
				get
				{
					if (m_DR[6] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDouble(m_DR[6]);
				}
			}
			private static object [] a_Z = new object[ArraySize];
			/// <summary>
			/// Retrieves Z for the current row. The return value can be System.DBNull.Value or a value that can be cast to double.
			/// </summary>
			public object _Z
			{
				get
				{
					if (m_DR[7] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDouble(m_DR[7]);
				}
			}
			private static object [] a_AREA = new object[ArraySize];
			/// <summary>
			/// Retrieves AREA for the current row. The return value can be System.DBNull.Value or a value that can be cast to int.
			/// </summary>
			public object _AREA
			{
				get
				{
					if (m_DR[8] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt32(m_DR[8]);
				}
			}
			private static object [] a_DARKNESS = new object[ArraySize];
			/// <summary>
			/// Retrieves DARKNESS for the current row. The return value can be System.DBNull.Value or a value that can be cast to double.
			/// </summary>
			public object _DARKNESS
			{
				get
				{
					if (m_DR[9] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDouble(m_DR[9]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is actually inserted later, in a batch insertion command. The Flush method must be called to ensure all rows are actually written.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">the value to be inserted for ID_EVENTBRICK. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_ID_ZONE">the value to be inserted for ID_ZONE. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_SIDE">the value to be inserted for SIDE. The value for this parameter can be int or System.DBNull.Value.</param>
			/// <param name="i_ID_MIPMICROTRACK">the value to be inserted for ID_MIPMICROTRACK. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_ID">the value to be inserted for ID. The value for this parameter can be double or System.DBNull.Value.</param>
			/// <param name="i_X">the value to be inserted for X. The value for this parameter can be double or System.DBNull.Value.</param>
			/// <param name="i_Y">the value to be inserted for Y. The value for this parameter can be double or System.DBNull.Value.</param>
			/// <param name="i_Z">the value to be inserted for Z. The value for this parameter can be double or System.DBNull.Value.</param>
			/// <param name="i_AREA">the value to be inserted for AREA. The value for this parameter can be int or System.DBNull.Value.</param>
			/// <param name="i_DARKNESS">the value to be inserted for DARKNESS. The value for this parameter can be double or System.DBNull.Value.</param>
			static public void Insert(object i_ID_EVENTBRICK,object i_ID_ZONE,object i_SIDE,object i_ID_MIPMICROTRACK,object i_ID,object i_X,object i_Y,object i_Z,object i_AREA,object i_DARKNESS)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				a_ID_EVENTBRICK[index] = i_ID_EVENTBRICK;
				a_ID_ZONE[index] = i_ID_ZONE;
				a_SIDE[index] = i_SIDE;
				a_ID_MIPMICROTRACK[index] = i_ID_MIPMICROTRACK;
				a_ID[index] = i_ID;
				a_X[index] = i_X;
				a_Y[index] = i_Y;
				a_Z[index] = i_Z;
				a_AREA[index] = i_AREA;
				a_DARKNESS[index] = i_DARKNESS;
				if (++index >= ArraySize) { cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			}
			/// <summary>
			/// Reads a set of rows from LZ_GRAINS and retrieves them into a new LZ_GRAINS object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the LZ_GRAINS class that can be used to read the retrieved data.</returns>
			static public LZ_GRAINS SelectWhere(string wherestr, string orderstr)
			{
				LZ_GRAINS newobj = new LZ_GRAINS();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_EVENTBRICK,ID_ZONE,SIDE,ID_MIPMICROTRACK,ID,X,Y,Z,AREA,DARKNESS FROM LZ_GRAINS" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO LZ_GRAINS (ID_EVENTBRICK,ID_ZONE,SIDE,ID_MIPMICROTRACK,ID,X,Y,Z,AREA,DARKNESS) VALUES (:p_1,:p_2,:p_3,:p_4,:p_5,:p_6,:p_7,:p_8,:p_9,:p_10)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_EVENTBRICK;
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_ZONE;
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Int, System.Data.ParameterDirection.Input).Value = a_SIDE;
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_MIPMICROTRACK;
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_ID;
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_X;
				newcmd.Parameters.Add("p_7", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_Y;
				newcmd.Parameters.Add("p_8", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_Z;
				newcmd.Parameters.Add("p_9", SySal.OperaDb.OperaDbType.Int, System.Data.ParameterDirection.Input).Value = a_AREA;
				newcmd.Parameters.Add("p_10", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_DARKNESS;
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the LZ_MACHINEVARS table in the DB.
		/// For data insertion, the Insert and Flush static methods are used. This class inserts data in batches.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class LZ_MACHINEVARS
		{
			internal LZ_MACHINEVARS() {}
			System.Data.DataRowCollection m_DRC;
			const int ArraySize = 100;
			static int index = 0;
			/// <summary>
			/// Since this class uses batch insertion, this method must be called to ensure that all rows are actually inserted into LZ_MACHINEVARS. Failure to do so will result in incomplete writes.
			/// </summary>
			static public void Flush() { if (cmd == null || index == 0) return; cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			private static long [] a_ID_MACHINE = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_MACHINE for the current row.
			/// </summary>
			public long _ID_MACHINE
			{
				get
				{
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			private static string [] a_NAME = new string[ArraySize];
			/// <summary>
			/// Retrieves NAME for the current row.
			/// </summary>
			public string _NAME
			{
				get
				{
					return System.Convert.ToString(m_DR[1]);
				}
			}
			private static string [] a_VALUE = new string[ArraySize];
			/// <summary>
			/// Retrieves VALUE for the current row.
			/// </summary>
			public string _VALUE
			{
				get
				{
					return System.Convert.ToString(m_DR[2]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is actually inserted later, in a batch insertion command. The Flush method must be called to ensure all rows are actually written.
			/// </summary>
			/// <param name="i_ID_MACHINE">the value to be inserted for ID_MACHINE.</param>
			/// <param name="i_NAME">the value to be inserted for NAME.</param>
			/// <param name="i_VALUE">the value to be inserted for VALUE.</param>
			static public void Insert(long i_ID_MACHINE,string i_NAME,string i_VALUE)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				a_ID_MACHINE[index] = i_ID_MACHINE;
				a_NAME[index] = i_NAME;
				a_VALUE[index] = i_VALUE;
				if (++index >= ArraySize) { cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			}
			/// <summary>
			/// Reads a set of rows from LZ_MACHINEVARS and retrieves them into a new LZ_MACHINEVARS object.
			/// </summary>
			/// <param name="i_ID_MACHINE">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_NAME">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="order">the ordering scheme to be applied to returned rows. This applies to all columns in the primary key.</param>
			/// <returns>a new instance of the LZ_MACHINEVARS class that can be used to read the retrieved data.</returns>
			static public LZ_MACHINEVARS SelectPrimaryKey(object i_ID_MACHINE,object i_NAME, OrderBy order)
			{
				string wherestr = "";
				string wtempstr = "";
				if (i_ID_MACHINE != null)
				{
					if (i_ID_MACHINE == System.DBNull.Value) wtempstr = "ID_MACHINE IS NULL";
					else wtempstr = "ID_MACHINE = " + i_ID_MACHINE.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_NAME != null)
				{
					if (i_NAME == System.DBNull.Value) wtempstr = "NAME IS NULL";
					else wtempstr = "NAME = " + i_NAME.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (order == OrderBy.Ascending) return SelectWhere(wherestr, "ID_MACHINE ASC,NAME ASC");
				else if (order == OrderBy.Descending) return SelectWhere(wherestr, "ID_MACHINE DESC,NAME DESC");
				return SelectWhere(wherestr, null);
			}
			/// <summary>
			/// Reads a set of rows from LZ_MACHINEVARS and retrieves them into a new LZ_MACHINEVARS object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the LZ_MACHINEVARS class that can be used to read the retrieved data.</returns>
			static public LZ_MACHINEVARS SelectWhere(string wherestr, string orderstr)
			{
				LZ_MACHINEVARS newobj = new LZ_MACHINEVARS();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_MACHINE,NAME,VALUE FROM LZ_MACHINEVARS" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO LZ_MACHINEVARS (ID_MACHINE,NAME,VALUE) VALUES (:p_1,:p_2,:p_3)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_MACHINE;
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input).Value = a_NAME;
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input).Value = a_VALUE;
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the LZ_MIPBASETRACKS table in the DB.
		/// For data insertion, the Insert and Flush static methods are used. This class inserts data in batches.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class LZ_MIPBASETRACKS
		{
			internal LZ_MIPBASETRACKS() {}
			System.Data.DataRowCollection m_DRC;
			const int ArraySize = 100;
			static int index = 0;
			/// <summary>
			/// Since this class uses batch insertion, this method must be called to ensure that all rows are actually inserted into LZ_MIPBASETRACKS. Failure to do so will result in incomplete writes.
			/// </summary>
			static public void Flush() { if (cmd == null || index == 0) return; cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			private static object [] a_ID_EVENTBRICK = new object[ArraySize];
			/// <summary>
			/// Retrieves ID_EVENTBRICK for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _ID_EVENTBRICK
			{
				get
				{
					if (m_DR[0] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			private static object [] a_ID_ZONE = new object[ArraySize];
			/// <summary>
			/// Retrieves ID_ZONE for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _ID_ZONE
			{
				get
				{
					if (m_DR[1] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[1]);
				}
			}
			private static object [] a_ID = new object[ArraySize];
			/// <summary>
			/// Retrieves ID for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _ID
			{
				get
				{
					if (m_DR[2] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[2]);
				}
			}
			private static object [] a_POSX = new object[ArraySize];
			/// <summary>
			/// Retrieves POSX for the current row. The return value can be System.DBNull.Value or a value that can be cast to double.
			/// </summary>
			public object _POSX
			{
				get
				{
					if (m_DR[3] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDouble(m_DR[3]);
				}
			}
			private static object [] a_POSY = new object[ArraySize];
			/// <summary>
			/// Retrieves POSY for the current row. The return value can be System.DBNull.Value or a value that can be cast to double.
			/// </summary>
			public object _POSY
			{
				get
				{
					if (m_DR[4] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDouble(m_DR[4]);
				}
			}
			private static object [] a_SLOPEX = new object[ArraySize];
			/// <summary>
			/// Retrieves SLOPEX for the current row. The return value can be System.DBNull.Value or a value that can be cast to double.
			/// </summary>
			public object _SLOPEX
			{
				get
				{
					if (m_DR[5] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDouble(m_DR[5]);
				}
			}
			private static object [] a_SLOPEY = new object[ArraySize];
			/// <summary>
			/// Retrieves SLOPEY for the current row. The return value can be System.DBNull.Value or a value that can be cast to double.
			/// </summary>
			public object _SLOPEY
			{
				get
				{
					if (m_DR[6] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDouble(m_DR[6]);
				}
			}
			private static object [] a_GRAINS = new object[ArraySize];
			/// <summary>
			/// Retrieves GRAINS for the current row. The return value can be System.DBNull.Value or a value that can be cast to int.
			/// </summary>
			public object _GRAINS
			{
				get
				{
					if (m_DR[7] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt32(m_DR[7]);
				}
			}
			private static object [] a_AREASUM = new object[ArraySize];
			/// <summary>
			/// Retrieves AREASUM for the current row. The return value can be System.DBNull.Value or a value that can be cast to int.
			/// </summary>
			public object _AREASUM
			{
				get
				{
					if (m_DR[8] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt32(m_DR[8]);
				}
			}
			private static object [] a_PH = new object[ArraySize];
			/// <summary>
			/// Retrieves PH for the current row. The return value can be System.DBNull.Value or a value that can be cast to double.
			/// </summary>
			public object _PH
			{
				get
				{
					if (m_DR[9] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDouble(m_DR[9]);
				}
			}
			private static object [] a_SIGMA = new object[ArraySize];
			/// <summary>
			/// Retrieves SIGMA for the current row. The return value can be System.DBNull.Value or a value that can be cast to double.
			/// </summary>
			public object _SIGMA
			{
				get
				{
					if (m_DR[10] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDouble(m_DR[10]);
				}
			}
			private static object [] a_ID_DOWNSIDE = new object[ArraySize];
			/// <summary>
			/// Retrieves ID_DOWNSIDE for the current row. The return value can be System.DBNull.Value or a value that can be cast to int.
			/// </summary>
			public object _ID_DOWNSIDE
			{
				get
				{
					if (m_DR[11] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt32(m_DR[11]);
				}
			}
			private static object [] a_ID_DOWNTRACK = new object[ArraySize];
			/// <summary>
			/// Retrieves ID_DOWNTRACK for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _ID_DOWNTRACK
			{
				get
				{
					if (m_DR[12] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[12]);
				}
			}
			private static object [] a_ID_UPSIDE = new object[ArraySize];
			/// <summary>
			/// Retrieves ID_UPSIDE for the current row. The return value can be System.DBNull.Value or a value that can be cast to int.
			/// </summary>
			public object _ID_UPSIDE
			{
				get
				{
					if (m_DR[13] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt32(m_DR[13]);
				}
			}
			private static object [] a_ID_UPTRACK = new object[ArraySize];
			/// <summary>
			/// Retrieves ID_UPTRACK for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _ID_UPTRACK
			{
				get
				{
					if (m_DR[14] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[14]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is actually inserted later, in a batch insertion command. The Flush method must be called to ensure all rows are actually written.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">the value to be inserted for ID_EVENTBRICK. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_ID_ZONE">the value to be inserted for ID_ZONE. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_ID">the value to be inserted for ID. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_POSX">the value to be inserted for POSX. The value for this parameter can be double or System.DBNull.Value.</param>
			/// <param name="i_POSY">the value to be inserted for POSY. The value for this parameter can be double or System.DBNull.Value.</param>
			/// <param name="i_SLOPEX">the value to be inserted for SLOPEX. The value for this parameter can be double or System.DBNull.Value.</param>
			/// <param name="i_SLOPEY">the value to be inserted for SLOPEY. The value for this parameter can be double or System.DBNull.Value.</param>
			/// <param name="i_GRAINS">the value to be inserted for GRAINS. The value for this parameter can be int or System.DBNull.Value.</param>
			/// <param name="i_AREASUM">the value to be inserted for AREASUM. The value for this parameter can be int or System.DBNull.Value.</param>
			/// <param name="i_PH">the value to be inserted for PH. The value for this parameter can be double or System.DBNull.Value.</param>
			/// <param name="i_SIGMA">the value to be inserted for SIGMA. The value for this parameter can be double or System.DBNull.Value.</param>
			/// <param name="i_ID_DOWNSIDE">the value to be inserted for ID_DOWNSIDE. The value for this parameter can be int or System.DBNull.Value.</param>
			/// <param name="i_ID_DOWNTRACK">the value to be inserted for ID_DOWNTRACK. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_ID_UPSIDE">the value to be inserted for ID_UPSIDE. The value for this parameter can be int or System.DBNull.Value.</param>
			/// <param name="i_ID_UPTRACK">the value to be inserted for ID_UPTRACK. The value for this parameter can be long or System.DBNull.Value.</param>
			static public void Insert(object i_ID_EVENTBRICK,object i_ID_ZONE,object i_ID,object i_POSX,object i_POSY,object i_SLOPEX,object i_SLOPEY,object i_GRAINS,object i_AREASUM,object i_PH,object i_SIGMA,object i_ID_DOWNSIDE,object i_ID_DOWNTRACK,object i_ID_UPSIDE,object i_ID_UPTRACK)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				a_ID_EVENTBRICK[index] = i_ID_EVENTBRICK;
				a_ID_ZONE[index] = i_ID_ZONE;
				a_ID[index] = i_ID;
				a_POSX[index] = i_POSX;
				a_POSY[index] = i_POSY;
				a_SLOPEX[index] = i_SLOPEX;
				a_SLOPEY[index] = i_SLOPEY;
				a_GRAINS[index] = i_GRAINS;
				a_AREASUM[index] = i_AREASUM;
				a_PH[index] = i_PH;
				a_SIGMA[index] = i_SIGMA;
				a_ID_DOWNSIDE[index] = i_ID_DOWNSIDE;
				a_ID_DOWNTRACK[index] = i_ID_DOWNTRACK;
				a_ID_UPSIDE[index] = i_ID_UPSIDE;
				a_ID_UPTRACK[index] = i_ID_UPTRACK;
				if (++index >= ArraySize) { cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			}
			/// <summary>
			/// Reads a set of rows from LZ_MIPBASETRACKS and retrieves them into a new LZ_MIPBASETRACKS object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the LZ_MIPBASETRACKS class that can be used to read the retrieved data.</returns>
			static public LZ_MIPBASETRACKS SelectWhere(string wherestr, string orderstr)
			{
				LZ_MIPBASETRACKS newobj = new LZ_MIPBASETRACKS();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_EVENTBRICK,ID_ZONE,ID,POSX,POSY,SLOPEX,SLOPEY,GRAINS,AREASUM,PH,SIGMA,ID_DOWNSIDE,ID_DOWNTRACK,ID_UPSIDE,ID_UPTRACK FROM LZ_MIPBASETRACKS" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO LZ_MIPBASETRACKS (ID_EVENTBRICK,ID_ZONE,ID,POSX,POSY,SLOPEX,SLOPEY,GRAINS,AREASUM,PH,SIGMA,ID_DOWNSIDE,ID_DOWNTRACK,ID_UPSIDE,ID_UPTRACK) VALUES (:p_1,:p_2,:p_3,:p_4,:p_5,:p_6,:p_7,:p_8,:p_9,:p_10,:p_11,:p_12,:p_13,:p_14,:p_15)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_EVENTBRICK;
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_ZONE;
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID;
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_POSX;
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_POSY;
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_SLOPEX;
				newcmd.Parameters.Add("p_7", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_SLOPEY;
				newcmd.Parameters.Add("p_8", SySal.OperaDb.OperaDbType.Int, System.Data.ParameterDirection.Input).Value = a_GRAINS;
				newcmd.Parameters.Add("p_9", SySal.OperaDb.OperaDbType.Int, System.Data.ParameterDirection.Input).Value = a_AREASUM;
				newcmd.Parameters.Add("p_10", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_PH;
				newcmd.Parameters.Add("p_11", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_SIGMA;
				newcmd.Parameters.Add("p_12", SySal.OperaDb.OperaDbType.Int, System.Data.ParameterDirection.Input).Value = a_ID_DOWNSIDE;
				newcmd.Parameters.Add("p_13", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_DOWNTRACK;
				newcmd.Parameters.Add("p_14", SySal.OperaDb.OperaDbType.Int, System.Data.ParameterDirection.Input).Value = a_ID_UPSIDE;
				newcmd.Parameters.Add("p_15", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_UPTRACK;
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the LZ_MIPMICROTRACKS table in the DB.
		/// For data insertion, the Insert and Flush static methods are used. This class inserts data in batches.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class LZ_MIPMICROTRACKS
		{
			internal LZ_MIPMICROTRACKS() {}
			System.Data.DataRowCollection m_DRC;
			const int ArraySize = 100;
			static int index = 0;
			/// <summary>
			/// Since this class uses batch insertion, this method must be called to ensure that all rows are actually inserted into LZ_MIPMICROTRACKS. Failure to do so will result in incomplete writes.
			/// </summary>
			static public void Flush() { if (cmd == null || index == 0) return; cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			private static object [] a_ID_EVENTBRICK = new object[ArraySize];
			/// <summary>
			/// Retrieves ID_EVENTBRICK for the current row. The return value can be System.DBNull.Value or a value that can be cast to double.
			/// </summary>
			public object _ID_EVENTBRICK
			{
				get
				{
					if (m_DR[0] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDouble(m_DR[0]);
				}
			}
			private static object [] a_ID_ZONE = new object[ArraySize];
			/// <summary>
			/// Retrieves ID_ZONE for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _ID_ZONE
			{
				get
				{
					if (m_DR[1] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[1]);
				}
			}
			private static object [] a_SIDE = new object[ArraySize];
			/// <summary>
			/// Retrieves SIDE for the current row. The return value can be System.DBNull.Value or a value that can be cast to int.
			/// </summary>
			public object _SIDE
			{
				get
				{
					if (m_DR[2] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt32(m_DR[2]);
				}
			}
			private static object [] a_ID = new object[ArraySize];
			/// <summary>
			/// Retrieves ID for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _ID
			{
				get
				{
					if (m_DR[3] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[3]);
				}
			}
			private static object [] a_POSX = new object[ArraySize];
			/// <summary>
			/// Retrieves POSX for the current row. The return value can be System.DBNull.Value or a value that can be cast to double.
			/// </summary>
			public object _POSX
			{
				get
				{
					if (m_DR[4] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDouble(m_DR[4]);
				}
			}
			private static object [] a_POSY = new object[ArraySize];
			/// <summary>
			/// Retrieves POSY for the current row. The return value can be System.DBNull.Value or a value that can be cast to double.
			/// </summary>
			public object _POSY
			{
				get
				{
					if (m_DR[5] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDouble(m_DR[5]);
				}
			}
			private static object [] a_SLOPEX = new object[ArraySize];
			/// <summary>
			/// Retrieves SLOPEX for the current row. The return value can be System.DBNull.Value or a value that can be cast to double.
			/// </summary>
			public object _SLOPEX
			{
				get
				{
					if (m_DR[6] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDouble(m_DR[6]);
				}
			}
			private static object [] a_SLOPEY = new object[ArraySize];
			/// <summary>
			/// Retrieves SLOPEY for the current row. The return value can be System.DBNull.Value or a value that can be cast to double.
			/// </summary>
			public object _SLOPEY
			{
				get
				{
					if (m_DR[7] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDouble(m_DR[7]);
				}
			}
			private static object [] a_GRAINS = new object[ArraySize];
			/// <summary>
			/// Retrieves GRAINS for the current row. The return value can be System.DBNull.Value or a value that can be cast to int.
			/// </summary>
			public object _GRAINS
			{
				get
				{
					if (m_DR[8] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt32(m_DR[8]);
				}
			}
			private static object [] a_AREASUM = new object[ArraySize];
			/// <summary>
			/// Retrieves AREASUM for the current row. The return value can be System.DBNull.Value or a value that can be cast to int.
			/// </summary>
			public object _AREASUM
			{
				get
				{
					if (m_DR[9] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt32(m_DR[9]);
				}
			}
			private static object [] a_PH = new object[ArraySize];
			/// <summary>
			/// Retrieves PH for the current row. The return value can be System.DBNull.Value or a value that can be cast to double.
			/// </summary>
			public object _PH
			{
				get
				{
					if (m_DR[10] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDouble(m_DR[10]);
				}
			}
			private static object [] a_SIGMA = new object[ArraySize];
			/// <summary>
			/// Retrieves SIGMA for the current row. The return value can be System.DBNull.Value or a value that can be cast to double.
			/// </summary>
			public object _SIGMA
			{
				get
				{
					if (m_DR[11] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDouble(m_DR[11]);
				}
			}
			private static object [] a_ID_VIEW = new object[ArraySize];
			/// <summary>
			/// Retrieves ID_VIEW for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _ID_VIEW
			{
				get
				{
					if (m_DR[12] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[12]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is actually inserted later, in a batch insertion command. The Flush method must be called to ensure all rows are actually written.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">the value to be inserted for ID_EVENTBRICK. The value for this parameter can be double or System.DBNull.Value.</param>
			/// <param name="i_ID_ZONE">the value to be inserted for ID_ZONE. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_SIDE">the value to be inserted for SIDE. The value for this parameter can be int or System.DBNull.Value.</param>
			/// <param name="i_ID">the value to be inserted for ID. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_POSX">the value to be inserted for POSX. The value for this parameter can be double or System.DBNull.Value.</param>
			/// <param name="i_POSY">the value to be inserted for POSY. The value for this parameter can be double or System.DBNull.Value.</param>
			/// <param name="i_SLOPEX">the value to be inserted for SLOPEX. The value for this parameter can be double or System.DBNull.Value.</param>
			/// <param name="i_SLOPEY">the value to be inserted for SLOPEY. The value for this parameter can be double or System.DBNull.Value.</param>
			/// <param name="i_GRAINS">the value to be inserted for GRAINS. The value for this parameter can be int or System.DBNull.Value.</param>
			/// <param name="i_AREASUM">the value to be inserted for AREASUM. The value for this parameter can be int or System.DBNull.Value.</param>
			/// <param name="i_PH">the value to be inserted for PH. The value for this parameter can be double or System.DBNull.Value.</param>
			/// <param name="i_SIGMA">the value to be inserted for SIGMA. The value for this parameter can be double or System.DBNull.Value.</param>
			/// <param name="i_ID_VIEW">the value to be inserted for ID_VIEW. The value for this parameter can be long or System.DBNull.Value.</param>
			static public void Insert(object i_ID_EVENTBRICK,object i_ID_ZONE,object i_SIDE,object i_ID,object i_POSX,object i_POSY,object i_SLOPEX,object i_SLOPEY,object i_GRAINS,object i_AREASUM,object i_PH,object i_SIGMA,object i_ID_VIEW)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				a_ID_EVENTBRICK[index] = i_ID_EVENTBRICK;
				a_ID_ZONE[index] = i_ID_ZONE;
				a_SIDE[index] = i_SIDE;
				a_ID[index] = i_ID;
				a_POSX[index] = i_POSX;
				a_POSY[index] = i_POSY;
				a_SLOPEX[index] = i_SLOPEX;
				a_SLOPEY[index] = i_SLOPEY;
				a_GRAINS[index] = i_GRAINS;
				a_AREASUM[index] = i_AREASUM;
				a_PH[index] = i_PH;
				a_SIGMA[index] = i_SIGMA;
				a_ID_VIEW[index] = i_ID_VIEW;
				if (++index >= ArraySize) { cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			}
			/// <summary>
			/// Reads a set of rows from LZ_MIPMICROTRACKS and retrieves them into a new LZ_MIPMICROTRACKS object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the LZ_MIPMICROTRACKS class that can be used to read the retrieved data.</returns>
			static public LZ_MIPMICROTRACKS SelectWhere(string wherestr, string orderstr)
			{
				LZ_MIPMICROTRACKS newobj = new LZ_MIPMICROTRACKS();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_EVENTBRICK,ID_ZONE,SIDE,ID,POSX,POSY,SLOPEX,SLOPEY,GRAINS,AREASUM,PH,SIGMA,ID_VIEW FROM LZ_MIPMICROTRACKS" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO LZ_MIPMICROTRACKS (ID_EVENTBRICK,ID_ZONE,SIDE,ID,POSX,POSY,SLOPEX,SLOPEY,GRAINS,AREASUM,PH,SIGMA,ID_VIEW) VALUES (:p_1,:p_2,:p_3,:p_4,:p_5,:p_6,:p_7,:p_8,:p_9,:p_10,:p_11,:p_12,:p_13)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_ID_EVENTBRICK;
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_ZONE;
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Int, System.Data.ParameterDirection.Input).Value = a_SIDE;
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID;
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_POSX;
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_POSY;
				newcmd.Parameters.Add("p_7", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_SLOPEX;
				newcmd.Parameters.Add("p_8", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_SLOPEY;
				newcmd.Parameters.Add("p_9", SySal.OperaDb.OperaDbType.Int, System.Data.ParameterDirection.Input).Value = a_GRAINS;
				newcmd.Parameters.Add("p_10", SySal.OperaDb.OperaDbType.Int, System.Data.ParameterDirection.Input).Value = a_AREASUM;
				newcmd.Parameters.Add("p_11", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_PH;
				newcmd.Parameters.Add("p_12", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_SIGMA;
				newcmd.Parameters.Add("p_13", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_VIEW;
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the LZ_PATTERN_MATCH table in the DB.
		/// For data insertion, the Insert and Flush static methods are used. This class inserts data in batches.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class LZ_PATTERN_MATCH
		{
			internal LZ_PATTERN_MATCH() {}
			System.Data.DataRowCollection m_DRC;
			const int ArraySize = 100;
			static int index = 0;
			/// <summary>
			/// Since this class uses batch insertion, this method must be called to ensure that all rows are actually inserted into LZ_PATTERN_MATCH. Failure to do so will result in incomplete writes.
			/// </summary>
			static public void Flush() { if (cmd == null || index == 0) return; cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			private static object [] a_ID_EVENTBRICK = new object[ArraySize];
			/// <summary>
			/// Retrieves ID_EVENTBRICK for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _ID_EVENTBRICK
			{
				get
				{
					if (m_DR[0] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			private static object [] a_ID_PROCESSOPERATION = new object[ArraySize];
			/// <summary>
			/// Retrieves ID_PROCESSOPERATION for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _ID_PROCESSOPERATION
			{
				get
				{
					if (m_DR[1] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[1]);
				}
			}
			private static object [] a_ID_FIRSTZONE = new object[ArraySize];
			/// <summary>
			/// Retrieves ID_FIRSTZONE for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _ID_FIRSTZONE
			{
				get
				{
					if (m_DR[2] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[2]);
				}
			}
			private static object [] a_ID_SECONDZONE = new object[ArraySize];
			/// <summary>
			/// Retrieves ID_SECONDZONE for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _ID_SECONDZONE
			{
				get
				{
					if (m_DR[3] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[3]);
				}
			}
			private static object [] a_ID_INFIRSTZONE = new object[ArraySize];
			/// <summary>
			/// Retrieves ID_INFIRSTZONE for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _ID_INFIRSTZONE
			{
				get
				{
					if (m_DR[4] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[4]);
				}
			}
			private static object [] a_ID_INSECONDZONE = new object[ArraySize];
			/// <summary>
			/// Retrieves ID_INSECONDZONE for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _ID_INSECONDZONE
			{
				get
				{
					if (m_DR[5] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[5]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is actually inserted later, in a batch insertion command. The Flush method must be called to ensure all rows are actually written.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">the value to be inserted for ID_EVENTBRICK. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_ID_PROCESSOPERATION">the value to be inserted for ID_PROCESSOPERATION. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_ID_FIRSTZONE">the value to be inserted for ID_FIRSTZONE. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_ID_SECONDZONE">the value to be inserted for ID_SECONDZONE. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_ID_INFIRSTZONE">the value to be inserted for ID_INFIRSTZONE. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_ID_INSECONDZONE">the value to be inserted for ID_INSECONDZONE. The value for this parameter can be long or System.DBNull.Value.</param>
			static public void Insert(object i_ID_EVENTBRICK,object i_ID_PROCESSOPERATION,object i_ID_FIRSTZONE,object i_ID_SECONDZONE,object i_ID_INFIRSTZONE,object i_ID_INSECONDZONE)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				a_ID_EVENTBRICK[index] = i_ID_EVENTBRICK;
				a_ID_PROCESSOPERATION[index] = i_ID_PROCESSOPERATION;
				a_ID_FIRSTZONE[index] = i_ID_FIRSTZONE;
				a_ID_SECONDZONE[index] = i_ID_SECONDZONE;
				a_ID_INFIRSTZONE[index] = i_ID_INFIRSTZONE;
				a_ID_INSECONDZONE[index] = i_ID_INSECONDZONE;
				if (++index >= ArraySize) { cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			}
			/// <summary>
			/// Reads a set of rows from LZ_PATTERN_MATCH and retrieves them into a new LZ_PATTERN_MATCH object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the LZ_PATTERN_MATCH class that can be used to read the retrieved data.</returns>
			static public LZ_PATTERN_MATCH SelectWhere(string wherestr, string orderstr)
			{
				LZ_PATTERN_MATCH newobj = new LZ_PATTERN_MATCH();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_EVENTBRICK,ID_PROCESSOPERATION,ID_FIRSTZONE,ID_SECONDZONE,ID_INFIRSTZONE,ID_INSECONDZONE FROM LZ_PATTERN_MATCH" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO LZ_PATTERN_MATCH (ID_EVENTBRICK,ID_PROCESSOPERATION,ID_FIRSTZONE,ID_SECONDZONE,ID_INFIRSTZONE,ID_INSECONDZONE) VALUES (:p_1,:p_2,:p_3,:p_4,:p_5,:p_6)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_EVENTBRICK;
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_PROCESSOPERATION;
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_FIRSTZONE;
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_SECONDZONE;
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_INFIRSTZONE;
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_INSECONDZONE;
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the LZ_PUBLISHERS table in the DB.
		/// For data insertion, the Insert and Flush static methods are used. This class inserts data in batches.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class LZ_PUBLISHERS
		{
			internal LZ_PUBLISHERS() {}
			System.Data.DataRowCollection m_DRC;
			const int ArraySize = 100;
			static int index = 0;
			/// <summary>
			/// Since this class uses batch insertion, this method must be called to ensure that all rows are actually inserted into LZ_PUBLISHERS. Failure to do so will result in incomplete writes.
			/// </summary>
			static public void Flush() { if (cmd == null || index == 0) return; cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			private static string [] a_NAME = new string[ArraySize];
			/// <summary>
			/// Retrieves NAME for the current row.
			/// </summary>
			public string _NAME
			{
				get
				{
					return System.Convert.ToString(m_DR[0]);
				}
			}
			private static string [] a_TYPE = new string[ArraySize];
			/// <summary>
			/// Retrieves TYPE for the current row.
			/// </summary>
			public string _TYPE
			{
				get
				{
					return System.Convert.ToString(m_DR[1]);
				}
			}
			private static object [] a_SPACEMIN = new object[ArraySize];
			/// <summary>
			/// Retrieves SPACEMIN for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _SPACEMIN
			{
				get
				{
					if (m_DR[2] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[2]);
				}
			}
			private static object [] a_SPACEMAX = new object[ArraySize];
			/// <summary>
			/// Retrieves SPACEMAX for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _SPACEMAX
			{
				get
				{
					if (m_DR[3] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[3]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is actually inserted later, in a batch insertion command. The Flush method must be called to ensure all rows are actually written.
			/// </summary>
			/// <param name="i_NAME">the value to be inserted for NAME.</param>
			/// <param name="i_TYPE">the value to be inserted for TYPE.</param>
			/// <param name="i_SPACEMIN">the value to be inserted for SPACEMIN. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_SPACEMAX">the value to be inserted for SPACEMAX. The value for this parameter can be long or System.DBNull.Value.</param>
			static public void Insert(string i_NAME,string i_TYPE,object i_SPACEMIN,object i_SPACEMAX)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				a_NAME[index] = i_NAME;
				a_TYPE[index] = i_TYPE;
				a_SPACEMIN[index] = i_SPACEMIN;
				a_SPACEMAX[index] = i_SPACEMAX;
				if (++index >= ArraySize) { cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			}
			/// <summary>
			/// Reads a set of rows from LZ_PUBLISHERS and retrieves them into a new LZ_PUBLISHERS object.
			/// </summary>
			/// <param name="i_NAME">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="order">the ordering scheme to be applied to returned rows.</param>
			/// <returns>a new instance of the LZ_PUBLISHERS class that can be used to read the retrieved data.</returns>
			static public LZ_PUBLISHERS SelectPrimaryKey(object i_NAME, OrderBy order)
			{
				string wherestr = "";
				string wtempstr = "";
				if (i_NAME != null)
				{
					if (i_NAME == System.DBNull.Value) wtempstr = "NAME IS NULL";
					else wtempstr = "NAME = " + i_NAME.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (order == OrderBy.Ascending) return SelectWhere(wherestr, "NAME ASC");
				else if (order == OrderBy.Descending) return SelectWhere(wherestr, "NAME DESC");
				return SelectWhere(wherestr, null);
			}
			/// <summary>
			/// Reads a set of rows from LZ_PUBLISHERS and retrieves them into a new LZ_PUBLISHERS object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the LZ_PUBLISHERS class that can be used to read the retrieved data.</returns>
			static public LZ_PUBLISHERS SelectWhere(string wherestr, string orderstr)
			{
				LZ_PUBLISHERS newobj = new LZ_PUBLISHERS();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT NAME,TYPE,SPACEMIN,SPACEMAX FROM LZ_PUBLISHERS" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO LZ_PUBLISHERS (NAME,TYPE,SPACEMIN,SPACEMAX) VALUES (:p_1,:p_2,:p_3,:p_4)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input).Value = a_NAME;
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input).Value = a_TYPE;
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_SPACEMIN;
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_SPACEMAX;
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the LZ_SCANBACK_CANCEL_PATH table in the DB.
		/// For data insertion, the Insert and Flush static methods are used. This class inserts data in batches.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class LZ_SCANBACK_CANCEL_PATH
		{
			internal LZ_SCANBACK_CANCEL_PATH() {}
			System.Data.DataRowCollection m_DRC;
			const int ArraySize = 100;
			static int index = 0;
			/// <summary>
			/// Since this class uses batch insertion, this method must be called to ensure that all rows are actually inserted into LZ_SCANBACK_CANCEL_PATH. Failure to do so will result in incomplete writes.
			/// </summary>
			static public void Flush() { if (cmd == null || index == 0) return; cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			private static object [] a_P_BRICKID = new object[ArraySize];
			/// <summary>
			/// Retrieves P_BRICKID for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _P_BRICKID
			{
				get
				{
					if (m_DR[0] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			private static object [] a_P_PATHID = new object[ArraySize];
			/// <summary>
			/// Retrieves P_PATHID for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _P_PATHID
			{
				get
				{
					if (m_DR[1] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[1]);
				}
			}
			private static object [] a_P_PLATEID = new object[ArraySize];
			/// <summary>
			/// Retrieves P_PLATEID for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _P_PLATEID
			{
				get
				{
					if (m_DR[2] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[2]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is actually inserted later, in a batch insertion command. The Flush method must be called to ensure all rows are actually written.
			/// </summary>
			/// <param name="i_P_BRICKID">the value to be inserted for P_BRICKID. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_P_PATHID">the value to be inserted for P_PATHID. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_P_PLATEID">the value to be inserted for P_PLATEID. The value for this parameter can be long or System.DBNull.Value.</param>
			static public void Insert(object i_P_BRICKID,object i_P_PATHID,object i_P_PLATEID)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				a_P_BRICKID[index] = i_P_BRICKID;
				a_P_PATHID[index] = i_P_PATHID;
				a_P_PLATEID[index] = i_P_PLATEID;
				if (++index >= ArraySize) { cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			}
			/// <summary>
			/// Reads a set of rows from LZ_SCANBACK_CANCEL_PATH and retrieves them into a new LZ_SCANBACK_CANCEL_PATH object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the LZ_SCANBACK_CANCEL_PATH class that can be used to read the retrieved data.</returns>
			static public LZ_SCANBACK_CANCEL_PATH SelectWhere(string wherestr, string orderstr)
			{
				LZ_SCANBACK_CANCEL_PATH newobj = new LZ_SCANBACK_CANCEL_PATH();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT P_BRICKID,P_PATHID,P_PLATEID FROM LZ_SCANBACK_CANCEL_PATH" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO LZ_SCANBACK_CANCEL_PATH (P_BRICKID,P_PATHID,P_PLATEID) VALUES (:p_1,:p_2,:p_3)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_P_BRICKID;
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_P_PATHID;
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_P_PLATEID;
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the LZ_SCANBACK_CANDIDATE table in the DB.
		/// For data insertion, the Insert and Flush static methods are used. This class inserts data in batches.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class LZ_SCANBACK_CANDIDATE
		{
			internal LZ_SCANBACK_CANDIDATE() {}
			System.Data.DataRowCollection m_DRC;
			const int ArraySize = 100;
			static int index = 0;
			/// <summary>
			/// Since this class uses batch insertion, this method must be called to ensure that all rows are actually inserted into LZ_SCANBACK_CANDIDATE. Failure to do so will result in incomplete writes.
			/// </summary>
			static public void Flush() { if (cmd == null || index == 0) return; cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			private static object [] a_P_BRICKID = new object[ArraySize];
			/// <summary>
			/// Retrieves P_BRICKID for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _P_BRICKID
			{
				get
				{
					if (m_DR[0] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			private static object [] a_P_PLATEID = new object[ArraySize];
			/// <summary>
			/// Retrieves P_PLATEID for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _P_PLATEID
			{
				get
				{
					if (m_DR[1] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[1]);
				}
			}
			private static object [] a_P_PATHID = new object[ArraySize];
			/// <summary>
			/// Retrieves P_PATHID for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _P_PATHID
			{
				get
				{
					if (m_DR[2] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[2]);
				}
			}
			private static object [] a_P_ZONEID = new object[ArraySize];
			/// <summary>
			/// Retrieves P_ZONEID for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _P_ZONEID
			{
				get
				{
					if (m_DR[3] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[3]);
				}
			}
			private static object [] a_P_CANDID = new object[ArraySize];
			/// <summary>
			/// Retrieves P_CANDID for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _P_CANDID
			{
				get
				{
					if (m_DR[4] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[4]);
				}
			}
			private static object [] a_P_MANUAL = new object[ArraySize];
			/// <summary>
			/// Retrieves P_MANUAL for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _P_MANUAL
			{
				get
				{
					if (m_DR[5] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[5]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is actually inserted later, in a batch insertion command. The Flush method must be called to ensure all rows are actually written.
			/// </summary>
			/// <param name="i_P_BRICKID">the value to be inserted for P_BRICKID. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_P_PLATEID">the value to be inserted for P_PLATEID. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_P_PATHID">the value to be inserted for P_PATHID. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_P_ZONEID">the value to be inserted for P_ZONEID. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_P_CANDID">the value to be inserted for P_CANDID. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_P_MANUAL">the value to be inserted for P_MANUAL. The value for this parameter can be long or System.DBNull.Value.</param>
			static public void Insert(object i_P_BRICKID,object i_P_PLATEID,object i_P_PATHID,object i_P_ZONEID,object i_P_CANDID,object i_P_MANUAL)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				a_P_BRICKID[index] = i_P_BRICKID;
				a_P_PLATEID[index] = i_P_PLATEID;
				a_P_PATHID[index] = i_P_PATHID;
				a_P_ZONEID[index] = i_P_ZONEID;
				a_P_CANDID[index] = i_P_CANDID;
				a_P_MANUAL[index] = i_P_MANUAL;
				if (++index >= ArraySize) { cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			}
			/// <summary>
			/// Reads a set of rows from LZ_SCANBACK_CANDIDATE and retrieves them into a new LZ_SCANBACK_CANDIDATE object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the LZ_SCANBACK_CANDIDATE class that can be used to read the retrieved data.</returns>
			static public LZ_SCANBACK_CANDIDATE SelectWhere(string wherestr, string orderstr)
			{
				LZ_SCANBACK_CANDIDATE newobj = new LZ_SCANBACK_CANDIDATE();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT P_BRICKID,P_PLATEID,P_PATHID,P_ZONEID,P_CANDID,P_MANUAL FROM LZ_SCANBACK_CANDIDATE" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO LZ_SCANBACK_CANDIDATE (P_BRICKID,P_PLATEID,P_PATHID,P_ZONEID,P_CANDID,P_MANUAL) VALUES (:p_1,:p_2,:p_3,:p_4,:p_5,:p_6)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_P_BRICKID;
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_P_PLATEID;
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_P_PATHID;
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_P_ZONEID;
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_P_CANDID;
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_P_MANUAL;
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the LZ_SCANBACK_DAMAGEDZONE table in the DB.
		/// For data insertion, the Insert and Flush static methods are used. This class inserts data in batches.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class LZ_SCANBACK_DAMAGEDZONE
		{
			internal LZ_SCANBACK_DAMAGEDZONE() {}
			System.Data.DataRowCollection m_DRC;
			const int ArraySize = 100;
			static int index = 0;
			/// <summary>
			/// Since this class uses batch insertion, this method must be called to ensure that all rows are actually inserted into LZ_SCANBACK_DAMAGEDZONE. Failure to do so will result in incomplete writes.
			/// </summary>
			static public void Flush() { if (cmd == null || index == 0) return; cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			private static object [] a_P_BRICKID = new object[ArraySize];
			/// <summary>
			/// Retrieves P_BRICKID for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _P_BRICKID
			{
				get
				{
					if (m_DR[0] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			private static object [] a_P_PLATEID = new object[ArraySize];
			/// <summary>
			/// Retrieves P_PLATEID for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _P_PLATEID
			{
				get
				{
					if (m_DR[1] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[1]);
				}
			}
			private static object [] a_P_PATHID = new object[ArraySize];
			/// <summary>
			/// Retrieves P_PATHID for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _P_PATHID
			{
				get
				{
					if (m_DR[2] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[2]);
				}
			}
			private static object [] a_P_DAMAGE = new object[ArraySize];
			/// <summary>
			/// Retrieves P_DAMAGE for the current row. The return value can be System.DBNull.Value or a value that can be cast to char.
			/// </summary>
			public object _P_DAMAGE
			{
				get
				{
					if (m_DR[3] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToChar(m_DR[3]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is actually inserted later, in a batch insertion command. The Flush method must be called to ensure all rows are actually written.
			/// </summary>
			/// <param name="i_P_BRICKID">the value to be inserted for P_BRICKID. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_P_PLATEID">the value to be inserted for P_PLATEID. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_P_PATHID">the value to be inserted for P_PATHID. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_P_DAMAGE">the value to be inserted for P_DAMAGE. The value for this parameter can be char or System.DBNull.Value.</param>
			static public void Insert(object i_P_BRICKID,object i_P_PLATEID,object i_P_PATHID,object i_P_DAMAGE)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				a_P_BRICKID[index] = i_P_BRICKID;
				a_P_PLATEID[index] = i_P_PLATEID;
				a_P_PATHID[index] = i_P_PATHID;
				a_P_DAMAGE[index] = i_P_DAMAGE;
				if (++index >= ArraySize) { cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			}
			/// <summary>
			/// Reads a set of rows from LZ_SCANBACK_DAMAGEDZONE and retrieves them into a new LZ_SCANBACK_DAMAGEDZONE object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the LZ_SCANBACK_DAMAGEDZONE class that can be used to read the retrieved data.</returns>
			static public LZ_SCANBACK_DAMAGEDZONE SelectWhere(string wherestr, string orderstr)
			{
				LZ_SCANBACK_DAMAGEDZONE newobj = new LZ_SCANBACK_DAMAGEDZONE();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT P_BRICKID,P_PLATEID,P_PATHID,P_DAMAGE FROM LZ_SCANBACK_DAMAGEDZONE" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO LZ_SCANBACK_DAMAGEDZONE (P_BRICKID,P_PLATEID,P_PATHID,P_DAMAGE) VALUES (:p_1,:p_2,:p_3,:p_4)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_P_BRICKID;
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_P_PLATEID;
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_P_PATHID;
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input).Value = a_P_DAMAGE;
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the LZ_SCANBACK_FORK table in the DB.
		/// For data insertion, the Insert and Flush static methods are used. This class inserts data in batches.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class LZ_SCANBACK_FORK
		{
			internal LZ_SCANBACK_FORK() {}
			System.Data.DataRowCollection m_DRC;
			const int ArraySize = 100;
			static int index = 0;
			/// <summary>
			/// Since this class uses batch insertion, this method must be called to ensure that all rows are actually inserted into LZ_SCANBACK_FORK. Failure to do so will result in incomplete writes.
			/// </summary>
			static public void Flush() { if (cmd == null || index == 0) return; cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			private static object [] a_P_BRICKID = new object[ArraySize];
			/// <summary>
			/// Retrieves P_BRICKID for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _P_BRICKID
			{
				get
				{
					if (m_DR[0] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			private static object [] a_P_PLATEID = new object[ArraySize];
			/// <summary>
			/// Retrieves P_PLATEID for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _P_PLATEID
			{
				get
				{
					if (m_DR[1] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[1]);
				}
			}
			private static object [] a_P_PATHID = new object[ArraySize];
			/// <summary>
			/// Retrieves P_PATHID for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _P_PATHID
			{
				get
				{
					if (m_DR[2] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[2]);
				}
			}
			private static object [] a_P_ZONEID = new object[ArraySize];
			/// <summary>
			/// Retrieves P_ZONEID for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _P_ZONEID
			{
				get
				{
					if (m_DR[3] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[3]);
				}
			}
			private static object [] a_P_CANDID = new object[ArraySize];
			/// <summary>
			/// Retrieves P_CANDID for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _P_CANDID
			{
				get
				{
					if (m_DR[4] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[4]);
				}
			}
			private static object [] a_P_FORKID = new object[ArraySize];
			/// <summary>
			/// Retrieves P_FORKID for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _P_FORKID
			{
				get
				{
					if (m_DR[5] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[5]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is actually inserted later, in a batch insertion command. The Flush method must be called to ensure all rows are actually written.
			/// </summary>
			/// <param name="i_P_BRICKID">the value to be inserted for P_BRICKID. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_P_PLATEID">the value to be inserted for P_PLATEID. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_P_PATHID">the value to be inserted for P_PATHID. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_P_ZONEID">the value to be inserted for P_ZONEID. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_P_CANDID">the value to be inserted for P_CANDID. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_P_FORKID">the value to be inserted for P_FORKID. The value for this parameter can be long or System.DBNull.Value.</param>
			static public void Insert(object i_P_BRICKID,object i_P_PLATEID,object i_P_PATHID,object i_P_ZONEID,object i_P_CANDID,object i_P_FORKID)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				a_P_BRICKID[index] = i_P_BRICKID;
				a_P_PLATEID[index] = i_P_PLATEID;
				a_P_PATHID[index] = i_P_PATHID;
				a_P_ZONEID[index] = i_P_ZONEID;
				a_P_CANDID[index] = i_P_CANDID;
				a_P_FORKID[index] = i_P_FORKID;
				if (++index >= ArraySize) { cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			}
			/// <summary>
			/// Reads a set of rows from LZ_SCANBACK_FORK and retrieves them into a new LZ_SCANBACK_FORK object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the LZ_SCANBACK_FORK class that can be used to read the retrieved data.</returns>
			static public LZ_SCANBACK_FORK SelectWhere(string wherestr, string orderstr)
			{
				LZ_SCANBACK_FORK newobj = new LZ_SCANBACK_FORK();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT P_BRICKID,P_PLATEID,P_PATHID,P_ZONEID,P_CANDID,P_FORKID FROM LZ_SCANBACK_FORK" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO LZ_SCANBACK_FORK (P_BRICKID,P_PLATEID,P_PATHID,P_ZONEID,P_CANDID,P_FORKID) VALUES (:p_1,:p_2,:p_3,:p_4,:p_5,:p_6)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_P_BRICKID;
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_P_PLATEID;
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_P_PATHID;
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_P_ZONEID;
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_P_CANDID;
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_P_FORKID;
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the LZ_SCANBACK_NOCANDIDATE table in the DB.
		/// For data insertion, the Insert and Flush static methods are used. This class inserts data in batches.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class LZ_SCANBACK_NOCANDIDATE
		{
			internal LZ_SCANBACK_NOCANDIDATE() {}
			System.Data.DataRowCollection m_DRC;
			const int ArraySize = 100;
			static int index = 0;
			/// <summary>
			/// Since this class uses batch insertion, this method must be called to ensure that all rows are actually inserted into LZ_SCANBACK_NOCANDIDATE. Failure to do so will result in incomplete writes.
			/// </summary>
			static public void Flush() { if (cmd == null || index == 0) return; cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			private static object [] a_P_BRICKID = new object[ArraySize];
			/// <summary>
			/// Retrieves P_BRICKID for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _P_BRICKID
			{
				get
				{
					if (m_DR[0] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			private static object [] a_P_PLATEID = new object[ArraySize];
			/// <summary>
			/// Retrieves P_PLATEID for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _P_PLATEID
			{
				get
				{
					if (m_DR[1] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[1]);
				}
			}
			private static object [] a_P_PATHID = new object[ArraySize];
			/// <summary>
			/// Retrieves P_PATHID for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _P_PATHID
			{
				get
				{
					if (m_DR[2] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[2]);
				}
			}
			private static object [] a_P_ZONEID = new object[ArraySize];
			/// <summary>
			/// Retrieves P_ZONEID for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _P_ZONEID
			{
				get
				{
					if (m_DR[3] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[3]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is actually inserted later, in a batch insertion command. The Flush method must be called to ensure all rows are actually written.
			/// </summary>
			/// <param name="i_P_BRICKID">the value to be inserted for P_BRICKID. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_P_PLATEID">the value to be inserted for P_PLATEID. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_P_PATHID">the value to be inserted for P_PATHID. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_P_ZONEID">the value to be inserted for P_ZONEID. The value for this parameter can be long or System.DBNull.Value.</param>
			static public void Insert(object i_P_BRICKID,object i_P_PLATEID,object i_P_PATHID,object i_P_ZONEID)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				a_P_BRICKID[index] = i_P_BRICKID;
				a_P_PLATEID[index] = i_P_PLATEID;
				a_P_PATHID[index] = i_P_PATHID;
				a_P_ZONEID[index] = i_P_ZONEID;
				if (++index >= ArraySize) { cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			}
			/// <summary>
			/// Reads a set of rows from LZ_SCANBACK_NOCANDIDATE and retrieves them into a new LZ_SCANBACK_NOCANDIDATE object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the LZ_SCANBACK_NOCANDIDATE class that can be used to read the retrieved data.</returns>
			static public LZ_SCANBACK_NOCANDIDATE SelectWhere(string wherestr, string orderstr)
			{
				LZ_SCANBACK_NOCANDIDATE newobj = new LZ_SCANBACK_NOCANDIDATE();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT P_BRICKID,P_PLATEID,P_PATHID,P_ZONEID FROM LZ_SCANBACK_NOCANDIDATE" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO LZ_SCANBACK_NOCANDIDATE (P_BRICKID,P_PLATEID,P_PATHID,P_ZONEID) VALUES (:p_1,:p_2,:p_3,:p_4)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_P_BRICKID;
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_P_PLATEID;
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_P_PATHID;
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_P_ZONEID;
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the LZ_SET_VOLUMESLICE_ZONE table in the DB.
		/// For data insertion, the Insert and Flush static methods are used. This class inserts data in batches.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class LZ_SET_VOLUMESLICE_ZONE
		{
			internal LZ_SET_VOLUMESLICE_ZONE() {}
			System.Data.DataRowCollection m_DRC;
			const int ArraySize = 100;
			static int index = 0;
			/// <summary>
			/// Since this class uses batch insertion, this method must be called to ensure that all rows are actually inserted into LZ_SET_VOLUMESLICE_ZONE. Failure to do so will result in incomplete writes.
			/// </summary>
			static public void Flush() { if (cmd == null || index == 0) return; cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			private static object [] a_P_BRICKID = new object[ArraySize];
			/// <summary>
			/// Retrieves P_BRICKID for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _P_BRICKID
			{
				get
				{
					if (m_DR[0] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			private static object [] a_P_PLATEID = new object[ArraySize];
			/// <summary>
			/// Retrieves P_PLATEID for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _P_PLATEID
			{
				get
				{
					if (m_DR[1] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[1]);
				}
			}
			private static object [] a_P_VOLID = new object[ArraySize];
			/// <summary>
			/// Retrieves P_VOLID for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _P_VOLID
			{
				get
				{
					if (m_DR[2] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[2]);
				}
			}
			private static object [] a_P_ZONEID = new object[ArraySize];
			/// <summary>
			/// Retrieves P_ZONEID for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _P_ZONEID
			{
				get
				{
					if (m_DR[3] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[3]);
				}
			}
			private static object [] a_P_DAMAGE = new object[ArraySize];
			/// <summary>
			/// Retrieves P_DAMAGE for the current row. The return value can be System.DBNull.Value or a value that can be cast to char.
			/// </summary>
			public object _P_DAMAGE
			{
				get
				{
					if (m_DR[4] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToChar(m_DR[4]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is actually inserted later, in a batch insertion command. The Flush method must be called to ensure all rows are actually written.
			/// </summary>
			/// <param name="i_P_BRICKID">the value to be inserted for P_BRICKID. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_P_PLATEID">the value to be inserted for P_PLATEID. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_P_VOLID">the value to be inserted for P_VOLID. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_P_ZONEID">the value to be inserted for P_ZONEID. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_P_DAMAGE">the value to be inserted for P_DAMAGE. The value for this parameter can be char or System.DBNull.Value.</param>
			static public void Insert(object i_P_BRICKID,object i_P_PLATEID,object i_P_VOLID,object i_P_ZONEID,object i_P_DAMAGE)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				a_P_BRICKID[index] = i_P_BRICKID;
				a_P_PLATEID[index] = i_P_PLATEID;
				a_P_VOLID[index] = i_P_VOLID;
				a_P_ZONEID[index] = i_P_ZONEID;
				a_P_DAMAGE[index] = i_P_DAMAGE;
				if (++index >= ArraySize) { cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			}
			/// <summary>
			/// Reads a set of rows from LZ_SET_VOLUMESLICE_ZONE and retrieves them into a new LZ_SET_VOLUMESLICE_ZONE object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the LZ_SET_VOLUMESLICE_ZONE class that can be used to read the retrieved data.</returns>
			static public LZ_SET_VOLUMESLICE_ZONE SelectWhere(string wherestr, string orderstr)
			{
				LZ_SET_VOLUMESLICE_ZONE newobj = new LZ_SET_VOLUMESLICE_ZONE();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT P_BRICKID,P_PLATEID,P_VOLID,P_ZONEID,P_DAMAGE FROM LZ_SET_VOLUMESLICE_ZONE" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO LZ_SET_VOLUMESLICE_ZONE (P_BRICKID,P_PLATEID,P_VOLID,P_ZONEID,P_DAMAGE) VALUES (:p_1,:p_2,:p_3,:p_4,:p_5)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_P_BRICKID;
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_P_PLATEID;
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_P_VOLID;
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_P_ZONEID;
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input).Value = a_P_DAMAGE;
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the LZ_SITEVARS table in the DB.
		/// For data insertion, the Insert and Flush static methods are used. This class inserts data in batches.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class LZ_SITEVARS
		{
			internal LZ_SITEVARS() {}
			System.Data.DataRowCollection m_DRC;
			const int ArraySize = 100;
			static int index = 0;
			/// <summary>
			/// Since this class uses batch insertion, this method must be called to ensure that all rows are actually inserted into LZ_SITEVARS. Failure to do so will result in incomplete writes.
			/// </summary>
			static public void Flush() { if (cmd == null || index == 0) return; cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			private static string [] a_NAME = new string[ArraySize];
			/// <summary>
			/// Retrieves NAME for the current row.
			/// </summary>
			public string _NAME
			{
				get
				{
					return System.Convert.ToString(m_DR[0]);
				}
			}
			private static string [] a_VALUE = new string[ArraySize];
			/// <summary>
			/// Retrieves VALUE for the current row.
			/// </summary>
			public string _VALUE
			{
				get
				{
					return System.Convert.ToString(m_DR[1]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is actually inserted later, in a batch insertion command. The Flush method must be called to ensure all rows are actually written.
			/// </summary>
			/// <param name="i_NAME">the value to be inserted for NAME.</param>
			/// <param name="i_VALUE">the value to be inserted for VALUE.</param>
			static public void Insert(string i_NAME,string i_VALUE)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				a_NAME[index] = i_NAME;
				a_VALUE[index] = i_VALUE;
				if (++index >= ArraySize) { cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			}
			/// <summary>
			/// Reads a set of rows from LZ_SITEVARS and retrieves them into a new LZ_SITEVARS object.
			/// </summary>
			/// <param name="i_NAME">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="order">the ordering scheme to be applied to returned rows.</param>
			/// <returns>a new instance of the LZ_SITEVARS class that can be used to read the retrieved data.</returns>
			static public LZ_SITEVARS SelectPrimaryKey(object i_NAME, OrderBy order)
			{
				string wherestr = "";
				string wtempstr = "";
				if (i_NAME != null)
				{
					if (i_NAME == System.DBNull.Value) wtempstr = "NAME IS NULL";
					else wtempstr = "NAME = " + i_NAME.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (order == OrderBy.Ascending) return SelectWhere(wherestr, "NAME ASC");
				else if (order == OrderBy.Descending) return SelectWhere(wherestr, "NAME DESC");
				return SelectWhere(wherestr, null);
			}
			/// <summary>
			/// Reads a set of rows from LZ_SITEVARS and retrieves them into a new LZ_SITEVARS object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the LZ_SITEVARS class that can be used to read the retrieved data.</returns>
			static public LZ_SITEVARS SelectWhere(string wherestr, string orderstr)
			{
				LZ_SITEVARS newobj = new LZ_SITEVARS();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT NAME,VALUE FROM LZ_SITEVARS" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO LZ_SITEVARS (NAME,VALUE) VALUES (:p_1,:p_2)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input).Value = a_NAME;
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input).Value = a_VALUE;
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the LZ_TOKENS table in the DB.
		/// For data insertion, the Insert and Flush static methods are used. This class inserts data in batches.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class LZ_TOKENS
		{
			internal LZ_TOKENS() {}
			System.Data.DataRowCollection m_DRC;
			const int ArraySize = 100;
			static int index = 0;
			/// <summary>
			/// Since this class uses batch insertion, this method must be called to ensure that all rows are actually inserted into LZ_TOKENS. Failure to do so will result in incomplete writes.
			/// </summary>
			static public void Flush() { if (cmd == null || index == 0) return; cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			private static string [] a_ID = new string[ArraySize];
			/// <summary>
			/// Retrieves ID for the current row.
			/// </summary>
			public string _ID
			{
				get
				{
					return System.Convert.ToString(m_DR[0]);
				}
			}
			private static long [] a_ID_USER = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_USER for the current row.
			/// </summary>
			public long _ID_USER
			{
				get
				{
					return System.Convert.ToInt64(m_DR[1]);
				}
			}
			private static int [] a_REQUESTSCAN = new int[ArraySize];
			/// <summary>
			/// Retrieves REQUESTSCAN for the current row.
			/// </summary>
			public int _REQUESTSCAN
			{
				get
				{
					return System.Convert.ToInt32(m_DR[2]);
				}
			}
			private static int [] a_REQUESTWEBANALYSIS = new int[ArraySize];
			/// <summary>
			/// Retrieves REQUESTWEBANALYSIS for the current row.
			/// </summary>
			public int _REQUESTWEBANALYSIS
			{
				get
				{
					return System.Convert.ToInt32(m_DR[3]);
				}
			}
			private static int [] a_REQUESTDATAPROCESSING = new int[ArraySize];
			/// <summary>
			/// Retrieves REQUESTDATAPROCESSING for the current row.
			/// </summary>
			public int _REQUESTDATAPROCESSING
			{
				get
				{
					return System.Convert.ToInt32(m_DR[4]);
				}
			}
			private static int [] a_REQUESTDATADOWNLOAD = new int[ArraySize];
			/// <summary>
			/// Retrieves REQUESTDATADOWNLOAD for the current row.
			/// </summary>
			public int _REQUESTDATADOWNLOAD
			{
				get
				{
					return System.Convert.ToInt32(m_DR[5]);
				}
			}
			private static int [] a_REQUESTPROCESSSTARTUP = new int[ArraySize];
			/// <summary>
			/// Retrieves REQUESTPROCESSSTARTUP for the current row.
			/// </summary>
			public int _REQUESTPROCESSSTARTUP
			{
				get
				{
					return System.Convert.ToInt32(m_DR[6]);
				}
			}
			private static int [] a_ADMINISTER = new int[ArraySize];
			/// <summary>
			/// Retrieves ADMINISTER for the current row.
			/// </summary>
			public int _ADMINISTER
			{
				get
				{
					return System.Convert.ToInt32(m_DR[7]);
				}
			}
			private static System.DateTime [] a_CREATIONTIME = new System.DateTime[ArraySize];
			/// <summary>
			/// Retrieves CREATIONTIME for the current row.
			/// </summary>
			public System.DateTime _CREATIONTIME
			{
				get
				{
					return System.Convert.ToDateTime(m_DR[8]);
				}
			}
			private static long [] a_ID_PROCESSOPERATION = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_PROCESSOPERATION for the current row.
			/// </summary>
			public long _ID_PROCESSOPERATION
			{
				get
				{
					return System.Convert.ToInt64(m_DR[9]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is actually inserted later, in a batch insertion command. The Flush method must be called to ensure all rows are actually written.
			/// </summary>
			/// <param name="i_ID">the value to be inserted for ID.</param>
			/// <param name="i_ID_USER">the value to be inserted for ID_USER.</param>
			/// <param name="i_REQUESTSCAN">the value to be inserted for REQUESTSCAN.</param>
			/// <param name="i_REQUESTWEBANALYSIS">the value to be inserted for REQUESTWEBANALYSIS.</param>
			/// <param name="i_REQUESTDATAPROCESSING">the value to be inserted for REQUESTDATAPROCESSING.</param>
			/// <param name="i_REQUESTDATADOWNLOAD">the value to be inserted for REQUESTDATADOWNLOAD.</param>
			/// <param name="i_REQUESTPROCESSSTARTUP">the value to be inserted for REQUESTPROCESSSTARTUP.</param>
			/// <param name="i_ADMINISTER">the value to be inserted for ADMINISTER.</param>
			/// <param name="i_CREATIONTIME">the value to be inserted for CREATIONTIME.</param>
			/// <param name="i_ID_PROCESSOPERATION">the value to be inserted for ID_PROCESSOPERATION.</param>
			static public void Insert(string i_ID,long i_ID_USER,int i_REQUESTSCAN,int i_REQUESTWEBANALYSIS,int i_REQUESTDATAPROCESSING,int i_REQUESTDATADOWNLOAD,int i_REQUESTPROCESSSTARTUP,int i_ADMINISTER,System.DateTime i_CREATIONTIME,long i_ID_PROCESSOPERATION)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				a_ID[index] = i_ID;
				a_ID_USER[index] = i_ID_USER;
				a_REQUESTSCAN[index] = i_REQUESTSCAN;
				a_REQUESTWEBANALYSIS[index] = i_REQUESTWEBANALYSIS;
				a_REQUESTDATAPROCESSING[index] = i_REQUESTDATAPROCESSING;
				a_REQUESTDATADOWNLOAD[index] = i_REQUESTDATADOWNLOAD;
				a_REQUESTPROCESSSTARTUP[index] = i_REQUESTPROCESSSTARTUP;
				a_ADMINISTER[index] = i_ADMINISTER;
				a_CREATIONTIME[index] = i_CREATIONTIME;
				a_ID_PROCESSOPERATION[index] = i_ID_PROCESSOPERATION;
				if (++index >= ArraySize) { cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			}
			/// <summary>
			/// Reads a set of rows from LZ_TOKENS and retrieves them into a new LZ_TOKENS object.
			/// </summary>
			/// <param name="i_ID">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_PROCESSOPERATION">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="order">the ordering scheme to be applied to returned rows. This applies to all columns in the primary key.</param>
			/// <returns>a new instance of the LZ_TOKENS class that can be used to read the retrieved data.</returns>
			static public LZ_TOKENS SelectPrimaryKey(object i_ID,object i_ID_PROCESSOPERATION, OrderBy order)
			{
				string wherestr = "";
				string wtempstr = "";
				if (i_ID != null)
				{
					if (i_ID == System.DBNull.Value) wtempstr = "ID IS NULL";
					else wtempstr = "ID = " + i_ID.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_PROCESSOPERATION != null)
				{
					if (i_ID_PROCESSOPERATION == System.DBNull.Value) wtempstr = "ID_PROCESSOPERATION IS NULL";
					else wtempstr = "ID_PROCESSOPERATION = " + i_ID_PROCESSOPERATION.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (order == OrderBy.Ascending) return SelectWhere(wherestr, "ID ASC,ID_PROCESSOPERATION ASC");
				else if (order == OrderBy.Descending) return SelectWhere(wherestr, "ID DESC,ID_PROCESSOPERATION DESC");
				return SelectWhere(wherestr, null);
			}
			/// <summary>
			/// Reads a set of rows from LZ_TOKENS and retrieves them into a new LZ_TOKENS object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the LZ_TOKENS class that can be used to read the retrieved data.</returns>
			static public LZ_TOKENS SelectWhere(string wherestr, string orderstr)
			{
				LZ_TOKENS newobj = new LZ_TOKENS();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID,ID_USER,REQUESTSCAN,REQUESTWEBANALYSIS,REQUESTDATAPROCESSING,REQUESTDATADOWNLOAD,REQUESTPROCESSSTARTUP,ADMINISTER,CREATIONTIME,ID_PROCESSOPERATION FROM LZ_TOKENS" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO LZ_TOKENS (ID,ID_USER,REQUESTSCAN,REQUESTWEBANALYSIS,REQUESTDATAPROCESSING,REQUESTDATADOWNLOAD,REQUESTPROCESSSTARTUP,ADMINISTER,CREATIONTIME,ID_PROCESSOPERATION) VALUES (:p_1,:p_2,:p_3,:p_4,:p_5,:p_6,:p_7,:p_8,:p_9,:p_10)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input).Value = a_ID;
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_USER;
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Int, System.Data.ParameterDirection.Input).Value = a_REQUESTSCAN;
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Int, System.Data.ParameterDirection.Input).Value = a_REQUESTWEBANALYSIS;
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Int, System.Data.ParameterDirection.Input).Value = a_REQUESTDATAPROCESSING;
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.Int, System.Data.ParameterDirection.Input).Value = a_REQUESTDATADOWNLOAD;
				newcmd.Parameters.Add("p_7", SySal.OperaDb.OperaDbType.Int, System.Data.ParameterDirection.Input).Value = a_REQUESTPROCESSSTARTUP;
				newcmd.Parameters.Add("p_8", SySal.OperaDb.OperaDbType.Int, System.Data.ParameterDirection.Input).Value = a_ADMINISTER;
				newcmd.Parameters.Add("p_9", SySal.OperaDb.OperaDbType.DateTime, System.Data.ParameterDirection.Input).Value = a_CREATIONTIME;
				newcmd.Parameters.Add("p_10", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_PROCESSOPERATION;
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the LZ_VIEWS table in the DB.
		/// For data insertion, the Insert and Flush static methods are used. This class inserts data in batches.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class LZ_VIEWS
		{
			internal LZ_VIEWS() {}
			System.Data.DataRowCollection m_DRC;
			const int ArraySize = 100;
			static int index = 0;
			/// <summary>
			/// Since this class uses batch insertion, this method must be called to ensure that all rows are actually inserted into LZ_VIEWS. Failure to do so will result in incomplete writes.
			/// </summary>
			static public void Flush() { if (cmd == null || index == 0) return; cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			private static object [] a_ID_EVENTBRICK = new object[ArraySize];
			/// <summary>
			/// Retrieves ID_EVENTBRICK for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _ID_EVENTBRICK
			{
				get
				{
					if (m_DR[0] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			private static object [] a_ID_ZONE = new object[ArraySize];
			/// <summary>
			/// Retrieves ID_ZONE for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _ID_ZONE
			{
				get
				{
					if (m_DR[1] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[1]);
				}
			}
			private static object [] a_SIDE = new object[ArraySize];
			/// <summary>
			/// Retrieves SIDE for the current row. The return value can be System.DBNull.Value or a value that can be cast to int.
			/// </summary>
			public object _SIDE
			{
				get
				{
					if (m_DR[2] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt32(m_DR[2]);
				}
			}
			private static object [] a_ID = new object[ArraySize];
			/// <summary>
			/// Retrieves ID for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _ID
			{
				get
				{
					if (m_DR[3] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[3]);
				}
			}
			private static object [] a_DOWNZ = new object[ArraySize];
			/// <summary>
			/// Retrieves DOWNZ for the current row. The return value can be System.DBNull.Value or a value that can be cast to double.
			/// </summary>
			public object _DOWNZ
			{
				get
				{
					if (m_DR[4] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDouble(m_DR[4]);
				}
			}
			private static object [] a_UPZ = new object[ArraySize];
			/// <summary>
			/// Retrieves UPZ for the current row. The return value can be System.DBNull.Value or a value that can be cast to double.
			/// </summary>
			public object _UPZ
			{
				get
				{
					if (m_DR[5] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDouble(m_DR[5]);
				}
			}
			private static object [] a_POSX = new object[ArraySize];
			/// <summary>
			/// Retrieves POSX for the current row. The return value can be System.DBNull.Value or a value that can be cast to double.
			/// </summary>
			public object _POSX
			{
				get
				{
					if (m_DR[6] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDouble(m_DR[6]);
				}
			}
			private static object [] a_POSY = new object[ArraySize];
			/// <summary>
			/// Retrieves POSY for the current row. The return value can be System.DBNull.Value or a value that can be cast to double.
			/// </summary>
			public object _POSY
			{
				get
				{
					if (m_DR[7] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDouble(m_DR[7]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is actually inserted later, in a batch insertion command. The Flush method must be called to ensure all rows are actually written.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">the value to be inserted for ID_EVENTBRICK. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_ID_ZONE">the value to be inserted for ID_ZONE. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_SIDE">the value to be inserted for SIDE. The value for this parameter can be int or System.DBNull.Value.</param>
			/// <param name="i_ID">the value to be inserted for ID. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_DOWNZ">the value to be inserted for DOWNZ. The value for this parameter can be double or System.DBNull.Value.</param>
			/// <param name="i_UPZ">the value to be inserted for UPZ. The value for this parameter can be double or System.DBNull.Value.</param>
			/// <param name="i_POSX">the value to be inserted for POSX. The value for this parameter can be double or System.DBNull.Value.</param>
			/// <param name="i_POSY">the value to be inserted for POSY. The value for this parameter can be double or System.DBNull.Value.</param>
			static public void Insert(object i_ID_EVENTBRICK,object i_ID_ZONE,object i_SIDE,object i_ID,object i_DOWNZ,object i_UPZ,object i_POSX,object i_POSY)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				a_ID_EVENTBRICK[index] = i_ID_EVENTBRICK;
				a_ID_ZONE[index] = i_ID_ZONE;
				a_SIDE[index] = i_SIDE;
				a_ID[index] = i_ID;
				a_DOWNZ[index] = i_DOWNZ;
				a_UPZ[index] = i_UPZ;
				a_POSX[index] = i_POSX;
				a_POSY[index] = i_POSY;
				if (++index >= ArraySize) { cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			}
			/// <summary>
			/// Reads a set of rows from LZ_VIEWS and retrieves them into a new LZ_VIEWS object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the LZ_VIEWS class that can be used to read the retrieved data.</returns>
			static public LZ_VIEWS SelectWhere(string wherestr, string orderstr)
			{
				LZ_VIEWS newobj = new LZ_VIEWS();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_EVENTBRICK,ID_ZONE,SIDE,ID,DOWNZ,UPZ,POSX,POSY FROM LZ_VIEWS" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO LZ_VIEWS (ID_EVENTBRICK,ID_ZONE,SIDE,ID,DOWNZ,UPZ,POSX,POSY) VALUES (:p_1,:p_2,:p_3,:p_4,:p_5,:p_6,:p_7,:p_8)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_EVENTBRICK;
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_ZONE;
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Int, System.Data.ParameterDirection.Input).Value = a_SIDE;
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID;
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_DOWNZ;
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_UPZ;
				newcmd.Parameters.Add("p_7", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_POSX;
				newcmd.Parameters.Add("p_8", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_POSY;
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the LZ_ZONES table in the DB.
		/// For data insertion, the Insert method is used. Rows are inserted one by one.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class LZ_ZONES
		{
			internal LZ_ZONES() {}
			System.Data.DataRowCollection m_DRC;
			/// <summary>
			/// Retrieves ID_EVENTBRICK for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _ID_EVENTBRICK
			{
				get
				{
					if (m_DR[0] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			/// <summary>
			/// Retrieves ID for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _ID
			{
				get
				{
					if (m_DR[1] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[1]);
				}
			}
			/// <summary>
			/// Retrieves ID_PLATE for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _ID_PLATE
			{
				get
				{
					if (m_DR[2] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[2]);
				}
			}
			/// <summary>
			/// Retrieves ID_PROCESSOPERATION for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _ID_PROCESSOPERATION
			{
				get
				{
					if (m_DR[3] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[3]);
				}
			}
			/// <summary>
			/// Retrieves MINX for the current row. The return value can be System.DBNull.Value or a value that can be cast to double.
			/// </summary>
			public object _MINX
			{
				get
				{
					if (m_DR[4] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDouble(m_DR[4]);
				}
			}
			/// <summary>
			/// Retrieves MAXX for the current row. The return value can be System.DBNull.Value or a value that can be cast to double.
			/// </summary>
			public object _MAXX
			{
				get
				{
					if (m_DR[5] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDouble(m_DR[5]);
				}
			}
			/// <summary>
			/// Retrieves MINY for the current row. The return value can be System.DBNull.Value or a value that can be cast to double.
			/// </summary>
			public object _MINY
			{
				get
				{
					if (m_DR[6] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDouble(m_DR[6]);
				}
			}
			/// <summary>
			/// Retrieves MAXY for the current row. The return value can be System.DBNull.Value or a value that can be cast to double.
			/// </summary>
			public object _MAXY
			{
				get
				{
					if (m_DR[7] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDouble(m_DR[7]);
				}
			}
			/// <summary>
			/// Retrieves RAWDATAPATH for the current row. The return value can be System.DBNull.Value or a value that can be cast to string.
			/// </summary>
			public object _RAWDATAPATH
			{
				get
				{
					if (m_DR[8] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToString(m_DR[8]);
				}
			}
			/// <summary>
			/// Retrieves STARTTIME for the current row. The return value can be System.DBNull.Value or a value that can be cast to System.DateTime.
			/// </summary>
			public object _STARTTIME
			{
				get
				{
					if (m_DR[9] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDateTime(m_DR[9]);
				}
			}
			/// <summary>
			/// Retrieves ENDTIME for the current row. The return value can be System.DBNull.Value or a value that can be cast to System.DateTime.
			/// </summary>
			public object _ENDTIME
			{
				get
				{
					if (m_DR[10] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDateTime(m_DR[10]);
				}
			}
			/// <summary>
			/// Retrieves SERIES for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _SERIES
			{
				get
				{
					if (m_DR[11] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[11]);
				}
			}
			/// <summary>
			/// Retrieves TXX for the current row. The return value can be System.DBNull.Value or a value that can be cast to double.
			/// </summary>
			public object _TXX
			{
				get
				{
					if (m_DR[12] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDouble(m_DR[12]);
				}
			}
			/// <summary>
			/// Retrieves TXY for the current row. The return value can be System.DBNull.Value or a value that can be cast to double.
			/// </summary>
			public object _TXY
			{
				get
				{
					if (m_DR[13] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDouble(m_DR[13]);
				}
			}
			/// <summary>
			/// Retrieves TYX for the current row. The return value can be System.DBNull.Value or a value that can be cast to double.
			/// </summary>
			public object _TYX
			{
				get
				{
					if (m_DR[14] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDouble(m_DR[14]);
				}
			}
			/// <summary>
			/// Retrieves TYY for the current row. The return value can be System.DBNull.Value or a value that can be cast to double.
			/// </summary>
			public object _TYY
			{
				get
				{
					if (m_DR[15] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDouble(m_DR[15]);
				}
			}
			/// <summary>
			/// Retrieves TDX for the current row. The return value can be System.DBNull.Value or a value that can be cast to double.
			/// </summary>
			public object _TDX
			{
				get
				{
					if (m_DR[16] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDouble(m_DR[16]);
				}
			}
			/// <summary>
			/// Retrieves TDY for the current row. The return value can be System.DBNull.Value or a value that can be cast to double.
			/// </summary>
			public object _TDY
			{
				get
				{
					if (m_DR[17] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDouble(m_DR[17]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is inserted immediately.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">the value to be inserted for ID_EVENTBRICK. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_ID">the value to be inserted for ID. The value for this parameter can be long or System.DBNull.Value. This value is actually used only if this method call is involved in data publication/replication, otherwise the actual value is generated by the OPERA DB and the supplied value is ignored.</param>
			/// <param name="i_ID_PLATE">the value to be inserted for ID_PLATE. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_ID_PROCESSOPERATION">the value to be inserted for ID_PROCESSOPERATION. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_MINX">the value to be inserted for MINX. The value for this parameter can be double or System.DBNull.Value.</param>
			/// <param name="i_MAXX">the value to be inserted for MAXX. The value for this parameter can be double or System.DBNull.Value.</param>
			/// <param name="i_MINY">the value to be inserted for MINY. The value for this parameter can be double or System.DBNull.Value.</param>
			/// <param name="i_MAXY">the value to be inserted for MAXY. The value for this parameter can be double or System.DBNull.Value.</param>
			/// <param name="i_RAWDATAPATH">the value to be inserted for RAWDATAPATH. The value for this parameter can be string or System.DBNull.Value.</param>
			/// <param name="i_STARTTIME">the value to be inserted for STARTTIME. The value for this parameter can be System.DateTime or System.DBNull.Value.</param>
			/// <param name="i_ENDTIME">the value to be inserted for ENDTIME. The value for this parameter can be System.DateTime or System.DBNull.Value.</param>
			/// <param name="i_SERIES">the value to be inserted for SERIES. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_TXX">the value to be inserted for TXX. The value for this parameter can be double or System.DBNull.Value.</param>
			/// <param name="i_TXY">the value to be inserted for TXY. The value for this parameter can be double or System.DBNull.Value.</param>
			/// <param name="i_TYX">the value to be inserted for TYX. The value for this parameter can be double or System.DBNull.Value.</param>
			/// <param name="i_TYY">the value to be inserted for TYY. The value for this parameter can be double or System.DBNull.Value.</param>
			/// <param name="i_TDX">the value to be inserted for TDX. The value for this parameter can be double or System.DBNull.Value.</param>
			/// <param name="i_TDY">the value to be inserted for TDY. The value for this parameter can be double or System.DBNull.Value.</param>
			/// <returns>the value of ID for the new row.</returns>
			static public long Insert(object i_ID_EVENTBRICK,object i_ID,object i_ID_PLATE,object i_ID_PROCESSOPERATION,object i_MINX,object i_MAXX,object i_MINY,object i_MAXY,object i_RAWDATAPATH,object i_STARTTIME,object i_ENDTIME,object i_SERIES,object i_TXX,object i_TXY,object i_TYX,object i_TYY,object i_TDX,object i_TDY)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = i_ID_EVENTBRICK;
				cmd.Parameters[1].Value = i_ID;
				cmd.Parameters[2].Value = i_ID_PLATE;
				cmd.Parameters[3].Value = i_ID_PROCESSOPERATION;
				cmd.Parameters[4].Value = i_MINX;
				cmd.Parameters[5].Value = i_MAXX;
				cmd.Parameters[6].Value = i_MINY;
				cmd.Parameters[7].Value = i_MAXY;
				cmd.Parameters[8].Value = i_RAWDATAPATH;
				cmd.Parameters[9].Value = i_STARTTIME;
				cmd.Parameters[10].Value = i_ENDTIME;
				cmd.Parameters[11].Value = i_SERIES;
				cmd.Parameters[12].Value = i_TXX;
				cmd.Parameters[13].Value = i_TXY;
				cmd.Parameters[14].Value = i_TYX;
				cmd.Parameters[15].Value = i_TYY;
				cmd.Parameters[16].Value = i_TDX;
				cmd.Parameters[17].Value = i_TDY;
				cmd.ExecuteNonQuery();
				return SySal.OperaDb.Convert.ToInt64(cmd.Parameters[18].Value);
			}
			/// <summary>
			/// Reads a set of rows from LZ_ZONES and retrieves them into a new LZ_ZONES object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the LZ_ZONES class that can be used to read the retrieved data.</returns>
			static public LZ_ZONES SelectWhere(string wherestr, string orderstr)
			{
				LZ_ZONES newobj = new LZ_ZONES();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_EVENTBRICK,ID,ID_PLATE,ID_PROCESSOPERATION,MINX,MAXX,MINY,MAXY,RAWDATAPATH,STARTTIME,ENDTIME,SERIES,TXX,TXY,TYX,TYY,TDX,TDY FROM LZ_ZONES" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO LZ_ZONES (ID_EVENTBRICK,ID,ID_PLATE,ID_PROCESSOPERATION,MINX,MAXX,MINY,MAXY,RAWDATAPATH,STARTTIME,ENDTIME,SERIES,TXX,TXY,TYX,TYY,TDX,TDY) VALUES (:p_1,:p_2,:p_3,:p_4,:p_5,:p_6,:p_7,:p_8,:p_9,:p_10,:p_11,:p_12,:p_13,:p_14,:p_15,:p_16,:p_17,:p_18) RETURNING ID INTO :o_18");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_7", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_8", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_9", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_10", SySal.OperaDb.OperaDbType.DateTime, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_11", SySal.OperaDb.OperaDbType.DateTime, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_12", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_13", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_14", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_15", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_16", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_17", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_18", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("o_18", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Output);
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the TB_ALIGNED_MIPMICROTRACKS table in the DB.
		/// For data insertion, the Insert and Flush static methods are used. This class inserts data in batches.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class TB_ALIGNED_MIPMICROTRACKS
		{
			internal TB_ALIGNED_MIPMICROTRACKS() {}
			System.Data.DataRowCollection m_DRC;
			const int ArraySize = 100;
			static int index = 0;
			/// <summary>
			/// Since this class uses batch insertion, this method must be called to ensure that all rows are actually inserted into TB_ALIGNED_MIPMICROTRACKS. Failure to do so will result in incomplete writes.
			/// </summary>
			static public void Flush() { if (cmd == null || index == 0) return; cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			private static long [] a_ID_EVENTBRICK = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_EVENTBRICK for the current row.
			/// </summary>
			public long _ID_EVENTBRICK
			{
				get
				{
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			private static long [] a_ID_RECONSTRUCTION = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_RECONSTRUCTION for the current row.
			/// </summary>
			public long _ID_RECONSTRUCTION
			{
				get
				{
					return System.Convert.ToInt64(m_DR[1]);
				}
			}
			private static long [] a_ID_PLATE = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_PLATE for the current row.
			/// </summary>
			public long _ID_PLATE
			{
				get
				{
					return System.Convert.ToInt64(m_DR[2]);
				}
			}
			private static long [] a_ID_ZONE = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_ZONE for the current row.
			/// </summary>
			public long _ID_ZONE
			{
				get
				{
					return System.Convert.ToInt64(m_DR[3]);
				}
			}
			private static int [] a_SIDE = new int[ArraySize];
			/// <summary>
			/// Retrieves SIDE for the current row.
			/// </summary>
			public int _SIDE
			{
				get
				{
					return System.Convert.ToInt32(m_DR[4]);
				}
			}
			private static long [] a_ID = new long[ArraySize];
			/// <summary>
			/// Retrieves ID for the current row.
			/// </summary>
			public long _ID
			{
				get
				{
					return System.Convert.ToInt64(m_DR[5]);
				}
			}
			private static double [] a_POSX = new double[ArraySize];
			/// <summary>
			/// Retrieves POSX for the current row.
			/// </summary>
			public double _POSX
			{
				get
				{
					return System.Convert.ToDouble(m_DR[6]);
				}
			}
			private static double [] a_POSY = new double[ArraySize];
			/// <summary>
			/// Retrieves POSY for the current row.
			/// </summary>
			public double _POSY
			{
				get
				{
					return System.Convert.ToDouble(m_DR[7]);
				}
			}
			private static double [] a_SLOPEX = new double[ArraySize];
			/// <summary>
			/// Retrieves SLOPEX for the current row.
			/// </summary>
			public double _SLOPEX
			{
				get
				{
					return System.Convert.ToDouble(m_DR[8]);
				}
			}
			private static double [] a_SLOPEY = new double[ArraySize];
			/// <summary>
			/// Retrieves SLOPEY for the current row.
			/// </summary>
			public double _SLOPEY
			{
				get
				{
					return System.Convert.ToDouble(m_DR[9]);
				}
			}
			private static double [] a_SIGMA = new double[ArraySize];
			/// <summary>
			/// Retrieves SIGMA for the current row.
			/// </summary>
			public double _SIGMA
			{
				get
				{
					return System.Convert.ToDouble(m_DR[10]);
				}
			}
			private static object [] a_GRAINS = new object[ArraySize];
			/// <summary>
			/// Retrieves GRAINS for the current row. The return value can be System.DBNull.Value or a value that can be cast to int.
			/// </summary>
			public object _GRAINS
			{
				get
				{
					if (m_DR[11] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt32(m_DR[11]);
				}
			}
			private static object [] a_AREASUM = new object[ArraySize];
			/// <summary>
			/// Retrieves AREASUM for the current row. The return value can be System.DBNull.Value or a value that can be cast to int.
			/// </summary>
			public object _AREASUM
			{
				get
				{
					if (m_DR[12] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt32(m_DR[12]);
				}
			}
			private static object [] a_PH = new object[ArraySize];
			/// <summary>
			/// Retrieves PH for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _PH
			{
				get
				{
					if (m_DR[13] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[13]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is actually inserted later, in a batch insertion command. The Flush method must be called to ensure all rows are actually written.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">the value to be inserted for ID_EVENTBRICK.</param>
			/// <param name="i_ID_RECONSTRUCTION">the value to be inserted for ID_RECONSTRUCTION.</param>
			/// <param name="i_ID_PLATE">the value to be inserted for ID_PLATE.</param>
			/// <param name="i_ID_ZONE">the value to be inserted for ID_ZONE.</param>
			/// <param name="i_SIDE">the value to be inserted for SIDE.</param>
			/// <param name="i_ID">the value to be inserted for ID.</param>
			/// <param name="i_POSX">the value to be inserted for POSX.</param>
			/// <param name="i_POSY">the value to be inserted for POSY.</param>
			/// <param name="i_SLOPEX">the value to be inserted for SLOPEX.</param>
			/// <param name="i_SLOPEY">the value to be inserted for SLOPEY.</param>
			/// <param name="i_SIGMA">the value to be inserted for SIGMA.</param>
			/// <param name="i_GRAINS">the value to be inserted for GRAINS. The value for this parameter can be int or System.DBNull.Value.</param>
			/// <param name="i_AREASUM">the value to be inserted for AREASUM. The value for this parameter can be int or System.DBNull.Value.</param>
			/// <param name="i_PH">the value to be inserted for PH. The value for this parameter can be long or System.DBNull.Value.</param>
			static public void Insert(long i_ID_EVENTBRICK,long i_ID_RECONSTRUCTION,long i_ID_PLATE,long i_ID_ZONE,int i_SIDE,long i_ID,double i_POSX,double i_POSY,double i_SLOPEX,double i_SLOPEY,double i_SIGMA,object i_GRAINS,object i_AREASUM,object i_PH)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				a_ID_EVENTBRICK[index] = i_ID_EVENTBRICK;
				a_ID_RECONSTRUCTION[index] = i_ID_RECONSTRUCTION;
				a_ID_PLATE[index] = i_ID_PLATE;
				a_ID_ZONE[index] = i_ID_ZONE;
				a_SIDE[index] = i_SIDE;
				a_ID[index] = i_ID;
				a_POSX[index] = i_POSX;
				a_POSY[index] = i_POSY;
				a_SLOPEX[index] = i_SLOPEX;
				a_SLOPEY[index] = i_SLOPEY;
				a_SIGMA[index] = i_SIGMA;
				a_GRAINS[index] = i_GRAINS;
				a_AREASUM[index] = i_AREASUM;
				a_PH[index] = i_PH;
				if (++index >= ArraySize) { cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			}
			/// <summary>
			/// Reads a set of rows from TB_ALIGNED_MIPMICROTRACKS and retrieves them into a new TB_ALIGNED_MIPMICROTRACKS object.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_RECONSTRUCTION">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_PLATE">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_ZONE">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_SIDE">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="order">the ordering scheme to be applied to returned rows. This applies to all columns in the primary key.</param>
			/// <returns>a new instance of the TB_ALIGNED_MIPMICROTRACKS class that can be used to read the retrieved data.</returns>
			static public TB_ALIGNED_MIPMICROTRACKS SelectPrimaryKey(object i_ID_EVENTBRICK,object i_ID_RECONSTRUCTION,object i_ID_PLATE,object i_ID_ZONE,object i_SIDE,object i_ID, OrderBy order)
			{
				string wherestr = "";
				string wtempstr = "";
				if (i_ID_EVENTBRICK != null)
				{
					if (i_ID_EVENTBRICK == System.DBNull.Value) wtempstr = "ID_EVENTBRICK IS NULL";
					else wtempstr = "ID_EVENTBRICK = " + i_ID_EVENTBRICK.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_RECONSTRUCTION != null)
				{
					if (i_ID_RECONSTRUCTION == System.DBNull.Value) wtempstr = "ID_RECONSTRUCTION IS NULL";
					else wtempstr = "ID_RECONSTRUCTION = " + i_ID_RECONSTRUCTION.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_PLATE != null)
				{
					if (i_ID_PLATE == System.DBNull.Value) wtempstr = "ID_PLATE IS NULL";
					else wtempstr = "ID_PLATE = " + i_ID_PLATE.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_ZONE != null)
				{
					if (i_ID_ZONE == System.DBNull.Value) wtempstr = "ID_ZONE IS NULL";
					else wtempstr = "ID_ZONE = " + i_ID_ZONE.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_SIDE != null)
				{
					if (i_SIDE == System.DBNull.Value) wtempstr = "SIDE IS NULL";
					else wtempstr = "SIDE = " + i_SIDE.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID != null)
				{
					if (i_ID == System.DBNull.Value) wtempstr = "ID IS NULL";
					else wtempstr = "ID = " + i_ID.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (order == OrderBy.Ascending) return SelectWhere(wherestr, "ID_EVENTBRICK ASC,ID_RECONSTRUCTION ASC,ID_PLATE ASC,ID_ZONE ASC,SIDE ASC,ID ASC");
				else if (order == OrderBy.Descending) return SelectWhere(wherestr, "ID_EVENTBRICK DESC,ID_RECONSTRUCTION DESC,ID_PLATE DESC,ID_ZONE DESC,SIDE DESC,ID DESC");
				return SelectWhere(wherestr, null);
			}
			/// <summary>
			/// Reads a set of rows from TB_ALIGNED_MIPMICROTRACKS and retrieves them into a new TB_ALIGNED_MIPMICROTRACKS object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the TB_ALIGNED_MIPMICROTRACKS class that can be used to read the retrieved data.</returns>
			static public TB_ALIGNED_MIPMICROTRACKS SelectWhere(string wherestr, string orderstr)
			{
				TB_ALIGNED_MIPMICROTRACKS newobj = new TB_ALIGNED_MIPMICROTRACKS();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_EVENTBRICK,ID_RECONSTRUCTION,ID_PLATE,ID_ZONE,SIDE,ID,POSX,POSY,SLOPEX,SLOPEY,SIGMA,GRAINS,AREASUM,PH FROM TB_ALIGNED_MIPMICROTRACKS" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO TB_ALIGNED_MIPMICROTRACKS (ID_EVENTBRICK,ID_RECONSTRUCTION,ID_PLATE,ID_ZONE,SIDE,ID,POSX,POSY,SLOPEX,SLOPEY,SIGMA,GRAINS,AREASUM,PH) VALUES (:p_1,:p_2,:p_3,:p_4,:p_5,:p_6,:p_7,:p_8,:p_9,:p_10,:p_11,:p_12,:p_13,:p_14)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_EVENTBRICK;
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_RECONSTRUCTION;
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_PLATE;
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_ZONE;
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Int, System.Data.ParameterDirection.Input).Value = a_SIDE;
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID;
				newcmd.Parameters.Add("p_7", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_POSX;
				newcmd.Parameters.Add("p_8", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_POSY;
				newcmd.Parameters.Add("p_9", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_SLOPEX;
				newcmd.Parameters.Add("p_10", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_SLOPEY;
				newcmd.Parameters.Add("p_11", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_SIGMA;
				newcmd.Parameters.Add("p_12", SySal.OperaDb.OperaDbType.Int, System.Data.ParameterDirection.Input).Value = a_GRAINS;
				newcmd.Parameters.Add("p_13", SySal.OperaDb.OperaDbType.Int, System.Data.ParameterDirection.Input).Value = a_AREASUM;
				newcmd.Parameters.Add("p_14", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_PH;
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the TB_ALIGNED_SIDES table in the DB.
		/// For data insertion, the Insert and Flush static methods are used. This class inserts data in batches.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class TB_ALIGNED_SIDES
		{
			internal TB_ALIGNED_SIDES() {}
			System.Data.DataRowCollection m_DRC;
			const int ArraySize = 100;
			static int index = 0;
			/// <summary>
			/// Since this class uses batch insertion, this method must be called to ensure that all rows are actually inserted into TB_ALIGNED_SIDES. Failure to do so will result in incomplete writes.
			/// </summary>
			static public void Flush() { if (cmd == null || index == 0) return; cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			private static long [] a_ID_EVENTBRICK = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_EVENTBRICK for the current row.
			/// </summary>
			public long _ID_EVENTBRICK
			{
				get
				{
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			private static long [] a_ID_RECONSTRUCTION = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_RECONSTRUCTION for the current row.
			/// </summary>
			public long _ID_RECONSTRUCTION
			{
				get
				{
					return System.Convert.ToInt64(m_DR[1]);
				}
			}
			private static long [] a_ID_PLATE = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_PLATE for the current row.
			/// </summary>
			public long _ID_PLATE
			{
				get
				{
					return System.Convert.ToInt64(m_DR[2]);
				}
			}
			private static int [] a_SIDE = new int[ArraySize];
			/// <summary>
			/// Retrieves SIDE for the current row.
			/// </summary>
			public int _SIDE
			{
				get
				{
					return System.Convert.ToInt32(m_DR[3]);
				}
			}
			private static double [] a_DOWNZ = new double[ArraySize];
			/// <summary>
			/// Retrieves DOWNZ for the current row.
			/// </summary>
			public double _DOWNZ
			{
				get
				{
					return System.Convert.ToDouble(m_DR[4]);
				}
			}
			private static double [] a_UPZ = new double[ArraySize];
			/// <summary>
			/// Retrieves UPZ for the current row.
			/// </summary>
			public double _UPZ
			{
				get
				{
					return System.Convert.ToDouble(m_DR[5]);
				}
			}
			private static double [] a_SDX = new double[ArraySize];
			/// <summary>
			/// Retrieves SDX for the current row.
			/// </summary>
			public double _SDX
			{
				get
				{
					return System.Convert.ToDouble(m_DR[6]);
				}
			}
			private static double [] a_SDY = new double[ArraySize];
			/// <summary>
			/// Retrieves SDY for the current row.
			/// </summary>
			public double _SDY
			{
				get
				{
					return System.Convert.ToDouble(m_DR[7]);
				}
			}
			private static double [] a_SMX = new double[ArraySize];
			/// <summary>
			/// Retrieves SMX for the current row.
			/// </summary>
			public double _SMX
			{
				get
				{
					return System.Convert.ToDouble(m_DR[8]);
				}
			}
			private static double [] a_SMY = new double[ArraySize];
			/// <summary>
			/// Retrieves SMY for the current row.
			/// </summary>
			public double _SMY
			{
				get
				{
					return System.Convert.ToDouble(m_DR[9]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is actually inserted later, in a batch insertion command. The Flush method must be called to ensure all rows are actually written.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">the value to be inserted for ID_EVENTBRICK.</param>
			/// <param name="i_ID_RECONSTRUCTION">the value to be inserted for ID_RECONSTRUCTION.</param>
			/// <param name="i_ID_PLATE">the value to be inserted for ID_PLATE.</param>
			/// <param name="i_SIDE">the value to be inserted for SIDE.</param>
			/// <param name="i_DOWNZ">the value to be inserted for DOWNZ.</param>
			/// <param name="i_UPZ">the value to be inserted for UPZ.</param>
			/// <param name="i_SDX">the value to be inserted for SDX.</param>
			/// <param name="i_SDY">the value to be inserted for SDY.</param>
			/// <param name="i_SMX">the value to be inserted for SMX.</param>
			/// <param name="i_SMY">the value to be inserted for SMY.</param>
			static public void Insert(long i_ID_EVENTBRICK,long i_ID_RECONSTRUCTION,long i_ID_PLATE,int i_SIDE,double i_DOWNZ,double i_UPZ,double i_SDX,double i_SDY,double i_SMX,double i_SMY)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				a_ID_EVENTBRICK[index] = i_ID_EVENTBRICK;
				a_ID_RECONSTRUCTION[index] = i_ID_RECONSTRUCTION;
				a_ID_PLATE[index] = i_ID_PLATE;
				a_SIDE[index] = i_SIDE;
				a_DOWNZ[index] = i_DOWNZ;
				a_UPZ[index] = i_UPZ;
				a_SDX[index] = i_SDX;
				a_SDY[index] = i_SDY;
				a_SMX[index] = i_SMX;
				a_SMY[index] = i_SMY;
				if (++index >= ArraySize) { cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			}
			/// <summary>
			/// Reads a set of rows from TB_ALIGNED_SIDES and retrieves them into a new TB_ALIGNED_SIDES object.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_RECONSTRUCTION">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_PLATE">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_SIDE">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="order">the ordering scheme to be applied to returned rows. This applies to all columns in the primary key.</param>
			/// <returns>a new instance of the TB_ALIGNED_SIDES class that can be used to read the retrieved data.</returns>
			static public TB_ALIGNED_SIDES SelectPrimaryKey(object i_ID_EVENTBRICK,object i_ID_RECONSTRUCTION,object i_ID_PLATE,object i_SIDE, OrderBy order)
			{
				string wherestr = "";
				string wtempstr = "";
				if (i_ID_EVENTBRICK != null)
				{
					if (i_ID_EVENTBRICK == System.DBNull.Value) wtempstr = "ID_EVENTBRICK IS NULL";
					else wtempstr = "ID_EVENTBRICK = " + i_ID_EVENTBRICK.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_RECONSTRUCTION != null)
				{
					if (i_ID_RECONSTRUCTION == System.DBNull.Value) wtempstr = "ID_RECONSTRUCTION IS NULL";
					else wtempstr = "ID_RECONSTRUCTION = " + i_ID_RECONSTRUCTION.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_PLATE != null)
				{
					if (i_ID_PLATE == System.DBNull.Value) wtempstr = "ID_PLATE IS NULL";
					else wtempstr = "ID_PLATE = " + i_ID_PLATE.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_SIDE != null)
				{
					if (i_SIDE == System.DBNull.Value) wtempstr = "SIDE IS NULL";
					else wtempstr = "SIDE = " + i_SIDE.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (order == OrderBy.Ascending) return SelectWhere(wherestr, "ID_EVENTBRICK ASC,ID_RECONSTRUCTION ASC,ID_PLATE ASC,SIDE ASC");
				else if (order == OrderBy.Descending) return SelectWhere(wherestr, "ID_EVENTBRICK DESC,ID_RECONSTRUCTION DESC,ID_PLATE DESC,SIDE DESC");
				return SelectWhere(wherestr, null);
			}
			/// <summary>
			/// Reads a set of rows from TB_ALIGNED_SIDES and retrieves them into a new TB_ALIGNED_SIDES object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the TB_ALIGNED_SIDES class that can be used to read the retrieved data.</returns>
			static public TB_ALIGNED_SIDES SelectWhere(string wherestr, string orderstr)
			{
				TB_ALIGNED_SIDES newobj = new TB_ALIGNED_SIDES();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_EVENTBRICK,ID_RECONSTRUCTION,ID_PLATE,SIDE,DOWNZ,UPZ,SDX,SDY,SMX,SMY FROM TB_ALIGNED_SIDES" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO TB_ALIGNED_SIDES (ID_EVENTBRICK,ID_RECONSTRUCTION,ID_PLATE,SIDE,DOWNZ,UPZ,SDX,SDY,SMX,SMY) VALUES (:p_1,:p_2,:p_3,:p_4,:p_5,:p_6,:p_7,:p_8,:p_9,:p_10)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_EVENTBRICK;
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_RECONSTRUCTION;
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_PLATE;
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Int, System.Data.ParameterDirection.Input).Value = a_SIDE;
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_DOWNZ;
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_UPZ;
				newcmd.Parameters.Add("p_7", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_SDX;
				newcmd.Parameters.Add("p_8", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_SDY;
				newcmd.Parameters.Add("p_9", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_SMX;
				newcmd.Parameters.Add("p_10", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_SMY;
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the TB_ALIGNED_SLICES table in the DB.
		/// For data insertion, the Insert and Flush static methods are used. This class inserts data in batches.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class TB_ALIGNED_SLICES
		{
			internal TB_ALIGNED_SLICES() {}
			System.Data.DataRowCollection m_DRC;
			const int ArraySize = 100;
			static int index = 0;
			/// <summary>
			/// Since this class uses batch insertion, this method must be called to ensure that all rows are actually inserted into TB_ALIGNED_SLICES. Failure to do so will result in incomplete writes.
			/// </summary>
			static public void Flush() { if (cmd == null || index == 0) return; cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			private static long [] a_ID_EVENTBRICK = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_EVENTBRICK for the current row.
			/// </summary>
			public long _ID_EVENTBRICK
			{
				get
				{
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			private static long [] a_ID_RECONSTRUCTION = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_RECONSTRUCTION for the current row.
			/// </summary>
			public long _ID_RECONSTRUCTION
			{
				get
				{
					return System.Convert.ToInt64(m_DR[1]);
				}
			}
			private static long [] a_ID_PLATE = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_PLATE for the current row.
			/// </summary>
			public long _ID_PLATE
			{
				get
				{
					return System.Convert.ToInt64(m_DR[2]);
				}
			}
			private static double [] a_MINX = new double[ArraySize];
			/// <summary>
			/// Retrieves MINX for the current row.
			/// </summary>
			public double _MINX
			{
				get
				{
					return System.Convert.ToDouble(m_DR[3]);
				}
			}
			private static double [] a_MAXX = new double[ArraySize];
			/// <summary>
			/// Retrieves MAXX for the current row.
			/// </summary>
			public double _MAXX
			{
				get
				{
					return System.Convert.ToDouble(m_DR[4]);
				}
			}
			private static double [] a_MINY = new double[ArraySize];
			/// <summary>
			/// Retrieves MINY for the current row.
			/// </summary>
			public double _MINY
			{
				get
				{
					return System.Convert.ToDouble(m_DR[5]);
				}
			}
			private static double [] a_MAXY = new double[ArraySize];
			/// <summary>
			/// Retrieves MAXY for the current row.
			/// </summary>
			public double _MAXY
			{
				get
				{
					return System.Convert.ToDouble(m_DR[6]);
				}
			}
			private static double [] a_REFX = new double[ArraySize];
			/// <summary>
			/// Retrieves REFX for the current row.
			/// </summary>
			public double _REFX
			{
				get
				{
					return System.Convert.ToDouble(m_DR[7]);
				}
			}
			private static double [] a_REFY = new double[ArraySize];
			/// <summary>
			/// Retrieves REFY for the current row.
			/// </summary>
			public double _REFY
			{
				get
				{
					return System.Convert.ToDouble(m_DR[8]);
				}
			}
			private static double [] a_REFZ = new double[ArraySize];
			/// <summary>
			/// Retrieves REFZ for the current row.
			/// </summary>
			public double _REFZ
			{
				get
				{
					return System.Convert.ToDouble(m_DR[9]);
				}
			}
			private static double [] a_DOWNZ = new double[ArraySize];
			/// <summary>
			/// Retrieves DOWNZ for the current row.
			/// </summary>
			public double _DOWNZ
			{
				get
				{
					return System.Convert.ToDouble(m_DR[10]);
				}
			}
			private static double [] a_UPZ = new double[ArraySize];
			/// <summary>
			/// Retrieves UPZ for the current row.
			/// </summary>
			public double _UPZ
			{
				get
				{
					return System.Convert.ToDouble(m_DR[11]);
				}
			}
			private static double [] a_TXX = new double[ArraySize];
			/// <summary>
			/// Retrieves TXX for the current row.
			/// </summary>
			public double _TXX
			{
				get
				{
					return System.Convert.ToDouble(m_DR[12]);
				}
			}
			private static double [] a_TXY = new double[ArraySize];
			/// <summary>
			/// Retrieves TXY for the current row.
			/// </summary>
			public double _TXY
			{
				get
				{
					return System.Convert.ToDouble(m_DR[13]);
				}
			}
			private static double [] a_TYX = new double[ArraySize];
			/// <summary>
			/// Retrieves TYX for the current row.
			/// </summary>
			public double _TYX
			{
				get
				{
					return System.Convert.ToDouble(m_DR[14]);
				}
			}
			private static double [] a_TYY = new double[ArraySize];
			/// <summary>
			/// Retrieves TYY for the current row.
			/// </summary>
			public double _TYY
			{
				get
				{
					return System.Convert.ToDouble(m_DR[15]);
				}
			}
			private static double [] a_TDX = new double[ArraySize];
			/// <summary>
			/// Retrieves TDX for the current row.
			/// </summary>
			public double _TDX
			{
				get
				{
					return System.Convert.ToDouble(m_DR[16]);
				}
			}
			private static double [] a_TDY = new double[ArraySize];
			/// <summary>
			/// Retrieves TDY for the current row.
			/// </summary>
			public double _TDY
			{
				get
				{
					return System.Convert.ToDouble(m_DR[17]);
				}
			}
			private static double [] a_TDZ = new double[ArraySize];
			/// <summary>
			/// Retrieves TDZ for the current row.
			/// </summary>
			public double _TDZ
			{
				get
				{
					return System.Convert.ToDouble(m_DR[18]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is actually inserted later, in a batch insertion command. The Flush method must be called to ensure all rows are actually written.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">the value to be inserted for ID_EVENTBRICK.</param>
			/// <param name="i_ID_RECONSTRUCTION">the value to be inserted for ID_RECONSTRUCTION.</param>
			/// <param name="i_ID_PLATE">the value to be inserted for ID_PLATE.</param>
			/// <param name="i_MINX">the value to be inserted for MINX.</param>
			/// <param name="i_MAXX">the value to be inserted for MAXX.</param>
			/// <param name="i_MINY">the value to be inserted for MINY.</param>
			/// <param name="i_MAXY">the value to be inserted for MAXY.</param>
			/// <param name="i_REFX">the value to be inserted for REFX.</param>
			/// <param name="i_REFY">the value to be inserted for REFY.</param>
			/// <param name="i_REFZ">the value to be inserted for REFZ.</param>
			/// <param name="i_DOWNZ">the value to be inserted for DOWNZ.</param>
			/// <param name="i_UPZ">the value to be inserted for UPZ.</param>
			/// <param name="i_TXX">the value to be inserted for TXX.</param>
			/// <param name="i_TXY">the value to be inserted for TXY.</param>
			/// <param name="i_TYX">the value to be inserted for TYX.</param>
			/// <param name="i_TYY">the value to be inserted for TYY.</param>
			/// <param name="i_TDX">the value to be inserted for TDX.</param>
			/// <param name="i_TDY">the value to be inserted for TDY.</param>
			/// <param name="i_TDZ">the value to be inserted for TDZ.</param>
			static public void Insert(long i_ID_EVENTBRICK,long i_ID_RECONSTRUCTION,long i_ID_PLATE,double i_MINX,double i_MAXX,double i_MINY,double i_MAXY,double i_REFX,double i_REFY,double i_REFZ,double i_DOWNZ,double i_UPZ,double i_TXX,double i_TXY,double i_TYX,double i_TYY,double i_TDX,double i_TDY,double i_TDZ)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				a_ID_EVENTBRICK[index] = i_ID_EVENTBRICK;
				a_ID_RECONSTRUCTION[index] = i_ID_RECONSTRUCTION;
				a_ID_PLATE[index] = i_ID_PLATE;
				a_MINX[index] = i_MINX;
				a_MAXX[index] = i_MAXX;
				a_MINY[index] = i_MINY;
				a_MAXY[index] = i_MAXY;
				a_REFX[index] = i_REFX;
				a_REFY[index] = i_REFY;
				a_REFZ[index] = i_REFZ;
				a_DOWNZ[index] = i_DOWNZ;
				a_UPZ[index] = i_UPZ;
				a_TXX[index] = i_TXX;
				a_TXY[index] = i_TXY;
				a_TYX[index] = i_TYX;
				a_TYY[index] = i_TYY;
				a_TDX[index] = i_TDX;
				a_TDY[index] = i_TDY;
				a_TDZ[index] = i_TDZ;
				if (++index >= ArraySize) { cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			}
			/// <summary>
			/// Reads a set of rows from TB_ALIGNED_SLICES and retrieves them into a new TB_ALIGNED_SLICES object.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_RECONSTRUCTION">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_PLATE">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="order">the ordering scheme to be applied to returned rows. This applies to all columns in the primary key.</param>
			/// <returns>a new instance of the TB_ALIGNED_SLICES class that can be used to read the retrieved data.</returns>
			static public TB_ALIGNED_SLICES SelectPrimaryKey(object i_ID_EVENTBRICK,object i_ID_RECONSTRUCTION,object i_ID_PLATE, OrderBy order)
			{
				string wherestr = "";
				string wtempstr = "";
				if (i_ID_EVENTBRICK != null)
				{
					if (i_ID_EVENTBRICK == System.DBNull.Value) wtempstr = "ID_EVENTBRICK IS NULL";
					else wtempstr = "ID_EVENTBRICK = " + i_ID_EVENTBRICK.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_RECONSTRUCTION != null)
				{
					if (i_ID_RECONSTRUCTION == System.DBNull.Value) wtempstr = "ID_RECONSTRUCTION IS NULL";
					else wtempstr = "ID_RECONSTRUCTION = " + i_ID_RECONSTRUCTION.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_PLATE != null)
				{
					if (i_ID_PLATE == System.DBNull.Value) wtempstr = "ID_PLATE IS NULL";
					else wtempstr = "ID_PLATE = " + i_ID_PLATE.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (order == OrderBy.Ascending) return SelectWhere(wherestr, "ID_EVENTBRICK ASC,ID_RECONSTRUCTION ASC,ID_PLATE ASC");
				else if (order == OrderBy.Descending) return SelectWhere(wherestr, "ID_EVENTBRICK DESC,ID_RECONSTRUCTION DESC,ID_PLATE DESC");
				return SelectWhere(wherestr, null);
			}
			/// <summary>
			/// Reads a set of rows from TB_ALIGNED_SLICES and retrieves them into a new TB_ALIGNED_SLICES object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the TB_ALIGNED_SLICES class that can be used to read the retrieved data.</returns>
			static public TB_ALIGNED_SLICES SelectWhere(string wherestr, string orderstr)
			{
				TB_ALIGNED_SLICES newobj = new TB_ALIGNED_SLICES();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_EVENTBRICK,ID_RECONSTRUCTION,ID_PLATE,MINX,MAXX,MINY,MAXY,REFX,REFY,REFZ,DOWNZ,UPZ,TXX,TXY,TYX,TYY,TDX,TDY,TDZ FROM TB_ALIGNED_SLICES" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO TB_ALIGNED_SLICES (ID_EVENTBRICK,ID_RECONSTRUCTION,ID_PLATE,MINX,MAXX,MINY,MAXY,REFX,REFY,REFZ,DOWNZ,UPZ,TXX,TXY,TYX,TYY,TDX,TDY,TDZ) VALUES (:p_1,:p_2,:p_3,:p_4,:p_5,:p_6,:p_7,:p_8,:p_9,:p_10,:p_11,:p_12,:p_13,:p_14,:p_15,:p_16,:p_17,:p_18,:p_19)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_EVENTBRICK;
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_RECONSTRUCTION;
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_PLATE;
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_MINX;
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_MAXX;
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_MINY;
				newcmd.Parameters.Add("p_7", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_MAXY;
				newcmd.Parameters.Add("p_8", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_REFX;
				newcmd.Parameters.Add("p_9", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_REFY;
				newcmd.Parameters.Add("p_10", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_REFZ;
				newcmd.Parameters.Add("p_11", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_DOWNZ;
				newcmd.Parameters.Add("p_12", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_UPZ;
				newcmd.Parameters.Add("p_13", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_TXX;
				newcmd.Parameters.Add("p_14", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_TXY;
				newcmd.Parameters.Add("p_15", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_TYX;
				newcmd.Parameters.Add("p_16", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_TYY;
				newcmd.Parameters.Add("p_17", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_TDX;
				newcmd.Parameters.Add("p_18", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_TDY;
				newcmd.Parameters.Add("p_19", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_TDZ;
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the TB_BRICK_SETS table in the DB.
		/// For data insertion, the Insert and Flush static methods are used. This class inserts data in batches.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class TB_BRICK_SETS
		{
			internal TB_BRICK_SETS() {}
			System.Data.DataRowCollection m_DRC;
			const int ArraySize = 100;
			static int index = 0;
			/// <summary>
			/// Since this class uses batch insertion, this method must be called to ensure that all rows are actually inserted into TB_BRICK_SETS. Failure to do so will result in incomplete writes.
			/// </summary>
			static public void Flush() { if (cmd == null || index == 0) return; cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			private static string [] a_ID = new string[ArraySize];
			/// <summary>
			/// Retrieves ID for the current row.
			/// </summary>
			public string _ID
			{
				get
				{
					return System.Convert.ToString(m_DR[0]);
				}
			}
			private static long [] a_IDRANGE_MIN = new long[ArraySize];
			/// <summary>
			/// Retrieves IDRANGE_MIN for the current row.
			/// </summary>
			public long _IDRANGE_MIN
			{
				get
				{
					return System.Convert.ToInt64(m_DR[1]);
				}
			}
			private static long [] a_IDRANGE_MAX = new long[ArraySize];
			/// <summary>
			/// Retrieves IDRANGE_MAX for the current row.
			/// </summary>
			public long _IDRANGE_MAX
			{
				get
				{
					return System.Convert.ToInt64(m_DR[2]);
				}
			}
			private static string [] a_ID_PARTITION = new string[ArraySize];
			/// <summary>
			/// Retrieves ID_PARTITION for the current row.
			/// </summary>
			public string _ID_PARTITION
			{
				get
				{
					return System.Convert.ToString(m_DR[3]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is actually inserted later, in a batch insertion command. The Flush method must be called to ensure all rows are actually written.
			/// </summary>
			/// <param name="i_ID">the value to be inserted for ID.</param>
			/// <param name="i_IDRANGE_MIN">the value to be inserted for IDRANGE_MIN.</param>
			/// <param name="i_IDRANGE_MAX">the value to be inserted for IDRANGE_MAX.</param>
			/// <param name="i_ID_PARTITION">the value to be inserted for ID_PARTITION.</param>
			static public void Insert(string i_ID,long i_IDRANGE_MIN,long i_IDRANGE_MAX,string i_ID_PARTITION)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				a_ID[index] = i_ID;
				a_IDRANGE_MIN[index] = i_IDRANGE_MIN;
				a_IDRANGE_MAX[index] = i_IDRANGE_MAX;
				a_ID_PARTITION[index] = i_ID_PARTITION;
				if (++index >= ArraySize) { cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			}
			/// <summary>
			/// Reads a set of rows from TB_BRICK_SETS and retrieves them into a new TB_BRICK_SETS object.
			/// </summary>
			/// <param name="i_ID">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="order">the ordering scheme to be applied to returned rows.</param>
			/// <returns>a new instance of the TB_BRICK_SETS class that can be used to read the retrieved data.</returns>
			static public TB_BRICK_SETS SelectPrimaryKey(object i_ID, OrderBy order)
			{
				string wherestr = "";
				string wtempstr = "";
				if (i_ID != null)
				{
					if (i_ID == System.DBNull.Value) wtempstr = "ID IS NULL";
					else wtempstr = "ID = " + i_ID.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (order == OrderBy.Ascending) return SelectWhere(wherestr, "ID ASC");
				else if (order == OrderBy.Descending) return SelectWhere(wherestr, "ID DESC");
				return SelectWhere(wherestr, null);
			}
			/// <summary>
			/// Reads a set of rows from TB_BRICK_SETS and retrieves them into a new TB_BRICK_SETS object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the TB_BRICK_SETS class that can be used to read the retrieved data.</returns>
			static public TB_BRICK_SETS SelectWhere(string wherestr, string orderstr)
			{
				TB_BRICK_SETS newobj = new TB_BRICK_SETS();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID,IDRANGE_MIN,IDRANGE_MAX,ID_PARTITION FROM TB_BRICK_SETS" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO TB_BRICK_SETS (ID,IDRANGE_MIN,IDRANGE_MAX,ID_PARTITION) VALUES (:p_1,:p_2,:p_3,:p_4)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input).Value = a_ID;
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_IDRANGE_MIN;
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_IDRANGE_MAX;
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input).Value = a_ID_PARTITION;
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the TB_B_CSCANDS_SBPATHS table in the DB.
		/// For data insertion, the Insert and Flush static methods are used. This class inserts data in batches.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class TB_B_CSCANDS_SBPATHS
		{
			internal TB_B_CSCANDS_SBPATHS() {}
			System.Data.DataRowCollection m_DRC;
			const int ArraySize = 100;
			static int index = 0;
			/// <summary>
			/// Since this class uses batch insertion, this method must be called to ensure that all rows are actually inserted into TB_B_CSCANDS_SBPATHS. Failure to do so will result in incomplete writes.
			/// </summary>
			static public void Flush() { if (cmd == null || index == 0) return; cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			private static long [] a_ID_CS_EVENTBRICK = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_CS_EVENTBRICK for the current row.
			/// </summary>
			public long _ID_CS_EVENTBRICK
			{
				get
				{
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			private static long [] a_ID_CANDIDATE = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_CANDIDATE for the current row.
			/// </summary>
			public long _ID_CANDIDATE
			{
				get
				{
					return System.Convert.ToInt64(m_DR[1]);
				}
			}
			private static long [] a_ID_EVENTBRICK = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_EVENTBRICK for the current row.
			/// </summary>
			public long _ID_EVENTBRICK
			{
				get
				{
					return System.Convert.ToInt64(m_DR[2]);
				}
			}
			private static object [] a_ID_SCANBACK_PROCOPID = new object[ArraySize];
			/// <summary>
			/// Retrieves ID_SCANBACK_PROCOPID for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _ID_SCANBACK_PROCOPID
			{
				get
				{
					if (m_DR[3] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[3]);
				}
			}
			private static object [] a_PATH = new object[ArraySize];
			/// <summary>
			/// Retrieves PATH for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _PATH
			{
				get
				{
					if (m_DR[4] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[4]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is actually inserted later, in a batch insertion command. The Flush method must be called to ensure all rows are actually written.
			/// </summary>
			/// <param name="i_ID_CS_EVENTBRICK">the value to be inserted for ID_CS_EVENTBRICK.</param>
			/// <param name="i_ID_CANDIDATE">the value to be inserted for ID_CANDIDATE.</param>
			/// <param name="i_ID_EVENTBRICK">the value to be inserted for ID_EVENTBRICK.</param>
			/// <param name="i_ID_SCANBACK_PROCOPID">the value to be inserted for ID_SCANBACK_PROCOPID. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_PATH">the value to be inserted for PATH. The value for this parameter can be long or System.DBNull.Value.</param>
			static public void Insert(long i_ID_CS_EVENTBRICK,long i_ID_CANDIDATE,long i_ID_EVENTBRICK,object i_ID_SCANBACK_PROCOPID,object i_PATH)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				a_ID_CS_EVENTBRICK[index] = i_ID_CS_EVENTBRICK;
				a_ID_CANDIDATE[index] = i_ID_CANDIDATE;
				a_ID_EVENTBRICK[index] = i_ID_EVENTBRICK;
				a_ID_SCANBACK_PROCOPID[index] = i_ID_SCANBACK_PROCOPID;
				a_PATH[index] = i_PATH;
				if (++index >= ArraySize) { cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			}
			/// <summary>
			/// Reads a set of rows from TB_B_CSCANDS_SBPATHS and retrieves them into a new TB_B_CSCANDS_SBPATHS object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the TB_B_CSCANDS_SBPATHS class that can be used to read the retrieved data.</returns>
			static public TB_B_CSCANDS_SBPATHS SelectWhere(string wherestr, string orderstr)
			{
				TB_B_CSCANDS_SBPATHS newobj = new TB_B_CSCANDS_SBPATHS();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_CS_EVENTBRICK,ID_CANDIDATE,ID_EVENTBRICK,ID_SCANBACK_PROCOPID,PATH FROM TB_B_CSCANDS_SBPATHS" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO TB_B_CSCANDS_SBPATHS (ID_CS_EVENTBRICK,ID_CANDIDATE,ID_EVENTBRICK,ID_SCANBACK_PROCOPID,PATH) VALUES (:p_1,:p_2,:p_3,:p_4,:p_5)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_CS_EVENTBRICK;
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_CANDIDATE;
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_EVENTBRICK;
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_SCANBACK_PROCOPID;
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_PATH;
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the TB_B_PREDTRACKS_CSCANDS table in the DB.
		/// For data insertion, the Insert and Flush static methods are used. This class inserts data in batches.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class TB_B_PREDTRACKS_CSCANDS
		{
			internal TB_B_PREDTRACKS_CSCANDS() {}
			System.Data.DataRowCollection m_DRC;
			const int ArraySize = 100;
			static int index = 0;
			/// <summary>
			/// Since this class uses batch insertion, this method must be called to ensure that all rows are actually inserted into TB_B_PREDTRACKS_CSCANDS. Failure to do so will result in incomplete writes.
			/// </summary>
			static public void Flush() { if (cmd == null || index == 0) return; cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			private static long [] a_ID_EVENTBRICK = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_EVENTBRICK for the current row.
			/// </summary>
			public long _ID_EVENTBRICK
			{
				get
				{
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			private static long [] a_ID_CANDIDATE = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_CANDIDATE for the current row.
			/// </summary>
			public long _ID_CANDIDATE
			{
				get
				{
					return System.Convert.ToInt64(m_DR[1]);
				}
			}
			private static long [] a_ID_EVENT = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_EVENT for the current row.
			/// </summary>
			public long _ID_EVENT
			{
				get
				{
					return System.Convert.ToInt64(m_DR[2]);
				}
			}
			private static long [] a_TRACK = new long[ArraySize];
			/// <summary>
			/// Retrieves TRACK for the current row.
			/// </summary>
			public long _TRACK
			{
				get
				{
					return System.Convert.ToInt64(m_DR[3]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is actually inserted later, in a batch insertion command. The Flush method must be called to ensure all rows are actually written.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">the value to be inserted for ID_EVENTBRICK.</param>
			/// <param name="i_ID_CANDIDATE">the value to be inserted for ID_CANDIDATE.</param>
			/// <param name="i_ID_EVENT">the value to be inserted for ID_EVENT.</param>
			/// <param name="i_TRACK">the value to be inserted for TRACK.</param>
			static public void Insert(long i_ID_EVENTBRICK,long i_ID_CANDIDATE,long i_ID_EVENT,long i_TRACK)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				a_ID_EVENTBRICK[index] = i_ID_EVENTBRICK;
				a_ID_CANDIDATE[index] = i_ID_CANDIDATE;
				a_ID_EVENT[index] = i_ID_EVENT;
				a_TRACK[index] = i_TRACK;
				if (++index >= ArraySize) { cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			}
			/// <summary>
			/// Reads a set of rows from TB_B_PREDTRACKS_CSCANDS and retrieves them into a new TB_B_PREDTRACKS_CSCANDS object.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_CANDIDATE">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_EVENT">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_TRACK">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="order">the ordering scheme to be applied to returned rows. This applies to all columns in the primary key.</param>
			/// <returns>a new instance of the TB_B_PREDTRACKS_CSCANDS class that can be used to read the retrieved data.</returns>
			static public TB_B_PREDTRACKS_CSCANDS SelectPrimaryKey(object i_ID_EVENTBRICK,object i_ID_CANDIDATE,object i_ID_EVENT,object i_TRACK, OrderBy order)
			{
				string wherestr = "";
				string wtempstr = "";
				if (i_ID_EVENTBRICK != null)
				{
					if (i_ID_EVENTBRICK == System.DBNull.Value) wtempstr = "ID_EVENTBRICK IS NULL";
					else wtempstr = "ID_EVENTBRICK = " + i_ID_EVENTBRICK.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_CANDIDATE != null)
				{
					if (i_ID_CANDIDATE == System.DBNull.Value) wtempstr = "ID_CANDIDATE IS NULL";
					else wtempstr = "ID_CANDIDATE = " + i_ID_CANDIDATE.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_EVENT != null)
				{
					if (i_ID_EVENT == System.DBNull.Value) wtempstr = "ID_EVENT IS NULL";
					else wtempstr = "ID_EVENT = " + i_ID_EVENT.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_TRACK != null)
				{
					if (i_TRACK == System.DBNull.Value) wtempstr = "TRACK IS NULL";
					else wtempstr = "TRACK = " + i_TRACK.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (order == OrderBy.Ascending) return SelectWhere(wherestr, "ID_EVENTBRICK ASC,ID_CANDIDATE ASC,ID_EVENT ASC,TRACK ASC");
				else if (order == OrderBy.Descending) return SelectWhere(wherestr, "ID_EVENTBRICK DESC,ID_CANDIDATE DESC,ID_EVENT DESC,TRACK DESC");
				return SelectWhere(wherestr, null);
			}
			/// <summary>
			/// Reads a set of rows from TB_B_PREDTRACKS_CSCANDS and retrieves them into a new TB_B_PREDTRACKS_CSCANDS object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the TB_B_PREDTRACKS_CSCANDS class that can be used to read the retrieved data.</returns>
			static public TB_B_PREDTRACKS_CSCANDS SelectWhere(string wherestr, string orderstr)
			{
				TB_B_PREDTRACKS_CSCANDS newobj = new TB_B_PREDTRACKS_CSCANDS();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_EVENTBRICK,ID_CANDIDATE,ID_EVENT,TRACK FROM TB_B_PREDTRACKS_CSCANDS" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO TB_B_PREDTRACKS_CSCANDS (ID_EVENTBRICK,ID_CANDIDATE,ID_EVENT,TRACK) VALUES (:p_1,:p_2,:p_3,:p_4)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_EVENTBRICK;
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_CANDIDATE;
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_EVENT;
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_TRACK;
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the TB_B_PREDTRACKS_SBPATHS table in the DB.
		/// For data insertion, the Insert and Flush static methods are used. This class inserts data in batches.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class TB_B_PREDTRACKS_SBPATHS
		{
			internal TB_B_PREDTRACKS_SBPATHS() {}
			System.Data.DataRowCollection m_DRC;
			const int ArraySize = 100;
			static int index = 0;
			/// <summary>
			/// Since this class uses batch insertion, this method must be called to ensure that all rows are actually inserted into TB_B_PREDTRACKS_SBPATHS. Failure to do so will result in incomplete writes.
			/// </summary>
			static public void Flush() { if (cmd == null || index == 0) return; cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			private static long [] a_ID_EVENT = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_EVENT for the current row.
			/// </summary>
			public long _ID_EVENT
			{
				get
				{
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			private static long [] a_TRACK = new long[ArraySize];
			/// <summary>
			/// Retrieves TRACK for the current row.
			/// </summary>
			public long _TRACK
			{
				get
				{
					return System.Convert.ToInt64(m_DR[1]);
				}
			}
			private static long [] a_ID_EVENTBRICK = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_EVENTBRICK for the current row.
			/// </summary>
			public long _ID_EVENTBRICK
			{
				get
				{
					return System.Convert.ToInt64(m_DR[2]);
				}
			}
			private static object [] a_ID_SCANBACK_PROCOPID = new object[ArraySize];
			/// <summary>
			/// Retrieves ID_SCANBACK_PROCOPID for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _ID_SCANBACK_PROCOPID
			{
				get
				{
					if (m_DR[3] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[3]);
				}
			}
			private static object [] a_PATH = new object[ArraySize];
			/// <summary>
			/// Retrieves PATH for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _PATH
			{
				get
				{
					if (m_DR[4] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[4]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is actually inserted later, in a batch insertion command. The Flush method must be called to ensure all rows are actually written.
			/// </summary>
			/// <param name="i_ID_EVENT">the value to be inserted for ID_EVENT.</param>
			/// <param name="i_TRACK">the value to be inserted for TRACK.</param>
			/// <param name="i_ID_EVENTBRICK">the value to be inserted for ID_EVENTBRICK.</param>
			/// <param name="i_ID_SCANBACK_PROCOPID">the value to be inserted for ID_SCANBACK_PROCOPID. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_PATH">the value to be inserted for PATH. The value for this parameter can be long or System.DBNull.Value.</param>
			static public void Insert(long i_ID_EVENT,long i_TRACK,long i_ID_EVENTBRICK,object i_ID_SCANBACK_PROCOPID,object i_PATH)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				a_ID_EVENT[index] = i_ID_EVENT;
				a_TRACK[index] = i_TRACK;
				a_ID_EVENTBRICK[index] = i_ID_EVENTBRICK;
				a_ID_SCANBACK_PROCOPID[index] = i_ID_SCANBACK_PROCOPID;
				a_PATH[index] = i_PATH;
				if (++index >= ArraySize) { cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			}
			/// <summary>
			/// Reads a set of rows from TB_B_PREDTRACKS_SBPATHS and retrieves them into a new TB_B_PREDTRACKS_SBPATHS object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the TB_B_PREDTRACKS_SBPATHS class that can be used to read the retrieved data.</returns>
			static public TB_B_PREDTRACKS_SBPATHS SelectWhere(string wherestr, string orderstr)
			{
				TB_B_PREDTRACKS_SBPATHS newobj = new TB_B_PREDTRACKS_SBPATHS();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_EVENT,TRACK,ID_EVENTBRICK,ID_SCANBACK_PROCOPID,PATH FROM TB_B_PREDTRACKS_SBPATHS" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO TB_B_PREDTRACKS_SBPATHS (ID_EVENT,TRACK,ID_EVENTBRICK,ID_SCANBACK_PROCOPID,PATH) VALUES (:p_1,:p_2,:p_3,:p_4,:p_5)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_EVENT;
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_TRACK;
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_EVENTBRICK;
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_SCANBACK_PROCOPID;
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_PATH;
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the TB_B_SBPATHS_SBPATHS table in the DB.
		/// For data insertion, the Insert and Flush static methods are used. This class inserts data in batches.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class TB_B_SBPATHS_SBPATHS
		{
			internal TB_B_SBPATHS_SBPATHS() {}
			System.Data.DataRowCollection m_DRC;
			const int ArraySize = 100;
			static int index = 0;
			/// <summary>
			/// Since this class uses batch insertion, this method must be called to ensure that all rows are actually inserted into TB_B_SBPATHS_SBPATHS. Failure to do so will result in incomplete writes.
			/// </summary>
			static public void Flush() { if (cmd == null || index == 0) return; cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			private static long [] a_ID_PARENT_EVENTBRICK = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_PARENT_EVENTBRICK for the current row.
			/// </summary>
			public long _ID_PARENT_EVENTBRICK
			{
				get
				{
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			private static long [] a_ID_PARENT_SB_PROCOPID = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_PARENT_SB_PROCOPID for the current row.
			/// </summary>
			public long _ID_PARENT_SB_PROCOPID
			{
				get
				{
					return System.Convert.ToInt64(m_DR[1]);
				}
			}
			private static long [] a_PARENT_PATH = new long[ArraySize];
			/// <summary>
			/// Retrieves PARENT_PATH for the current row.
			/// </summary>
			public long _PARENT_PATH
			{
				get
				{
					return System.Convert.ToInt64(m_DR[2]);
				}
			}
			private static long [] a_ID_EVENTBRICK = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_EVENTBRICK for the current row.
			/// </summary>
			public long _ID_EVENTBRICK
			{
				get
				{
					return System.Convert.ToInt64(m_DR[3]);
				}
			}
			private static object [] a_ID_SCANBACK_PROCOPID = new object[ArraySize];
			/// <summary>
			/// Retrieves ID_SCANBACK_PROCOPID for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _ID_SCANBACK_PROCOPID
			{
				get
				{
					if (m_DR[4] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[4]);
				}
			}
			private static object [] a_PATH = new object[ArraySize];
			/// <summary>
			/// Retrieves PATH for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _PATH
			{
				get
				{
					if (m_DR[5] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[5]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is actually inserted later, in a batch insertion command. The Flush method must be called to ensure all rows are actually written.
			/// </summary>
			/// <param name="i_ID_PARENT_EVENTBRICK">the value to be inserted for ID_PARENT_EVENTBRICK.</param>
			/// <param name="i_ID_PARENT_SB_PROCOPID">the value to be inserted for ID_PARENT_SB_PROCOPID.</param>
			/// <param name="i_PARENT_PATH">the value to be inserted for PARENT_PATH.</param>
			/// <param name="i_ID_EVENTBRICK">the value to be inserted for ID_EVENTBRICK.</param>
			/// <param name="i_ID_SCANBACK_PROCOPID">the value to be inserted for ID_SCANBACK_PROCOPID. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_PATH">the value to be inserted for PATH. The value for this parameter can be long or System.DBNull.Value.</param>
			static public void Insert(long i_ID_PARENT_EVENTBRICK,long i_ID_PARENT_SB_PROCOPID,long i_PARENT_PATH,long i_ID_EVENTBRICK,object i_ID_SCANBACK_PROCOPID,object i_PATH)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				a_ID_PARENT_EVENTBRICK[index] = i_ID_PARENT_EVENTBRICK;
				a_ID_PARENT_SB_PROCOPID[index] = i_ID_PARENT_SB_PROCOPID;
				a_PARENT_PATH[index] = i_PARENT_PATH;
				a_ID_EVENTBRICK[index] = i_ID_EVENTBRICK;
				a_ID_SCANBACK_PROCOPID[index] = i_ID_SCANBACK_PROCOPID;
				a_PATH[index] = i_PATH;
				if (++index >= ArraySize) { cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			}
			/// <summary>
			/// Reads a set of rows from TB_B_SBPATHS_SBPATHS and retrieves them into a new TB_B_SBPATHS_SBPATHS object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the TB_B_SBPATHS_SBPATHS class that can be used to read the retrieved data.</returns>
			static public TB_B_SBPATHS_SBPATHS SelectWhere(string wherestr, string orderstr)
			{
				TB_B_SBPATHS_SBPATHS newobj = new TB_B_SBPATHS_SBPATHS();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_PARENT_EVENTBRICK,ID_PARENT_SB_PROCOPID,PARENT_PATH,ID_EVENTBRICK,ID_SCANBACK_PROCOPID,PATH FROM TB_B_SBPATHS_SBPATHS" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO TB_B_SBPATHS_SBPATHS (ID_PARENT_EVENTBRICK,ID_PARENT_SB_PROCOPID,PARENT_PATH,ID_EVENTBRICK,ID_SCANBACK_PROCOPID,PATH) VALUES (:p_1,:p_2,:p_3,:p_4,:p_5,:p_6)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_PARENT_EVENTBRICK;
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_PARENT_SB_PROCOPID;
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_PARENT_PATH;
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_EVENTBRICK;
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_SCANBACK_PROCOPID;
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_PATH;
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the TB_B_SBPATHS_VOLUMES table in the DB.
		/// For data insertion, the Insert and Flush static methods are used. This class inserts data in batches.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class TB_B_SBPATHS_VOLUMES
		{
			internal TB_B_SBPATHS_VOLUMES() {}
			System.Data.DataRowCollection m_DRC;
			const int ArraySize = 100;
			static int index = 0;
			/// <summary>
			/// Since this class uses batch insertion, this method must be called to ensure that all rows are actually inserted into TB_B_SBPATHS_VOLUMES. Failure to do so will result in incomplete writes.
			/// </summary>
			static public void Flush() { if (cmd == null || index == 0) return; cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			private static long [] a_ID_EVENTBRICK = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_EVENTBRICK for the current row.
			/// </summary>
			public long _ID_EVENTBRICK
			{
				get
				{
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			private static long [] a_ID_SCANBACK_PROCOPID = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_SCANBACK_PROCOPID for the current row.
			/// </summary>
			public long _ID_SCANBACK_PROCOPID
			{
				get
				{
					return System.Convert.ToInt64(m_DR[1]);
				}
			}
			private static long [] a_PATH = new long[ArraySize];
			/// <summary>
			/// Retrieves PATH for the current row.
			/// </summary>
			public long _PATH
			{
				get
				{
					return System.Convert.ToInt64(m_DR[2]);
				}
			}
			private static object [] a_ID_VOLUMESCAN_PROCOPID = new object[ArraySize];
			/// <summary>
			/// Retrieves ID_VOLUMESCAN_PROCOPID for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _ID_VOLUMESCAN_PROCOPID
			{
				get
				{
					if (m_DR[3] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[3]);
				}
			}
			private static object [] a_VOLUME = new object[ArraySize];
			/// <summary>
			/// Retrieves VOLUME for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _VOLUME
			{
				get
				{
					if (m_DR[4] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[4]);
				}
			}
			private static long [] a_ID_PLATE = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_PLATE for the current row.
			/// </summary>
			public long _ID_PLATE
			{
				get
				{
					return System.Convert.ToInt64(m_DR[5]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is actually inserted later, in a batch insertion command. The Flush method must be called to ensure all rows are actually written.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">the value to be inserted for ID_EVENTBRICK.</param>
			/// <param name="i_ID_SCANBACK_PROCOPID">the value to be inserted for ID_SCANBACK_PROCOPID.</param>
			/// <param name="i_PATH">the value to be inserted for PATH.</param>
			/// <param name="i_ID_VOLUMESCAN_PROCOPID">the value to be inserted for ID_VOLUMESCAN_PROCOPID. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_VOLUME">the value to be inserted for VOLUME. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_ID_PLATE">the value to be inserted for ID_PLATE.</param>
			static public void Insert(long i_ID_EVENTBRICK,long i_ID_SCANBACK_PROCOPID,long i_PATH,object i_ID_VOLUMESCAN_PROCOPID,object i_VOLUME,long i_ID_PLATE)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				a_ID_EVENTBRICK[index] = i_ID_EVENTBRICK;
				a_ID_SCANBACK_PROCOPID[index] = i_ID_SCANBACK_PROCOPID;
				a_PATH[index] = i_PATH;
				a_ID_VOLUMESCAN_PROCOPID[index] = i_ID_VOLUMESCAN_PROCOPID;
				a_VOLUME[index] = i_VOLUME;
				a_ID_PLATE[index] = i_ID_PLATE;
				if (++index >= ArraySize) { cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			}
			/// <summary>
			/// Reads a set of rows from TB_B_SBPATHS_VOLUMES and retrieves them into a new TB_B_SBPATHS_VOLUMES object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the TB_B_SBPATHS_VOLUMES class that can be used to read the retrieved data.</returns>
			static public TB_B_SBPATHS_VOLUMES SelectWhere(string wherestr, string orderstr)
			{
				TB_B_SBPATHS_VOLUMES newobj = new TB_B_SBPATHS_VOLUMES();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_EVENTBRICK,ID_SCANBACK_PROCOPID,PATH,ID_VOLUMESCAN_PROCOPID,VOLUME,ID_PLATE FROM TB_B_SBPATHS_VOLUMES" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO TB_B_SBPATHS_VOLUMES (ID_EVENTBRICK,ID_SCANBACK_PROCOPID,PATH,ID_VOLUMESCAN_PROCOPID,VOLUME,ID_PLATE) VALUES (:p_1,:p_2,:p_3,:p_4,:p_5,:p_6)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_EVENTBRICK;
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_SCANBACK_PROCOPID;
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_PATH;
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_VOLUMESCAN_PROCOPID;
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_VOLUME;
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_PLATE;
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the TB_B_VOLTKS_SBPATHS table in the DB.
		/// For data insertion, the Insert and Flush static methods are used. This class inserts data in batches.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class TB_B_VOLTKS_SBPATHS
		{
			internal TB_B_VOLTKS_SBPATHS() {}
			System.Data.DataRowCollection m_DRC;
			const int ArraySize = 100;
			static int index = 0;
			/// <summary>
			/// Since this class uses batch insertion, this method must be called to ensure that all rows are actually inserted into TB_B_VOLTKS_SBPATHS. Failure to do so will result in incomplete writes.
			/// </summary>
			static public void Flush() { if (cmd == null || index == 0) return; cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			private static long [] a_ID_EVENTBRICK = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_EVENTBRICK for the current row.
			/// </summary>
			public long _ID_EVENTBRICK
			{
				get
				{
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			private static long [] a_ID_RECONSTRUCTION = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_RECONSTRUCTION for the current row.
			/// </summary>
			public long _ID_RECONSTRUCTION
			{
				get
				{
					return System.Convert.ToInt64(m_DR[1]);
				}
			}
			private static long [] a_ID_OPTION = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_OPTION for the current row.
			/// </summary>
			public long _ID_OPTION
			{
				get
				{
					return System.Convert.ToInt64(m_DR[2]);
				}
			}
			private static long [] a_ID_VOLUMETRACK = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_VOLUMETRACK for the current row.
			/// </summary>
			public long _ID_VOLUMETRACK
			{
				get
				{
					return System.Convert.ToInt64(m_DR[3]);
				}
			}
			private static string [] a_EXTENT = new string[ArraySize];
			/// <summary>
			/// Retrieves EXTENT for the current row.
			/// </summary>
			public string _EXTENT
			{
				get
				{
					return System.Convert.ToString(m_DR[4]);
				}
			}
			private static object [] a_ID_SCANBACK_PROCOPID = new object[ArraySize];
			/// <summary>
			/// Retrieves ID_SCANBACK_PROCOPID for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _ID_SCANBACK_PROCOPID
			{
				get
				{
					if (m_DR[5] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[5]);
				}
			}
			private static object [] a_PATH = new object[ArraySize];
			/// <summary>
			/// Retrieves PATH for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _PATH
			{
				get
				{
					if (m_DR[6] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[6]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is actually inserted later, in a batch insertion command. The Flush method must be called to ensure all rows are actually written.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">the value to be inserted for ID_EVENTBRICK.</param>
			/// <param name="i_ID_RECONSTRUCTION">the value to be inserted for ID_RECONSTRUCTION.</param>
			/// <param name="i_ID_OPTION">the value to be inserted for ID_OPTION.</param>
			/// <param name="i_ID_VOLUMETRACK">the value to be inserted for ID_VOLUMETRACK.</param>
			/// <param name="i_EXTENT">the value to be inserted for EXTENT.</param>
			/// <param name="i_ID_SCANBACK_PROCOPID">the value to be inserted for ID_SCANBACK_PROCOPID. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_PATH">the value to be inserted for PATH. The value for this parameter can be long or System.DBNull.Value.</param>
			static public void Insert(long i_ID_EVENTBRICK,long i_ID_RECONSTRUCTION,long i_ID_OPTION,long i_ID_VOLUMETRACK,string i_EXTENT,object i_ID_SCANBACK_PROCOPID,object i_PATH)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				a_ID_EVENTBRICK[index] = i_ID_EVENTBRICK;
				a_ID_RECONSTRUCTION[index] = i_ID_RECONSTRUCTION;
				a_ID_OPTION[index] = i_ID_OPTION;
				a_ID_VOLUMETRACK[index] = i_ID_VOLUMETRACK;
				a_EXTENT[index] = i_EXTENT;
				a_ID_SCANBACK_PROCOPID[index] = i_ID_SCANBACK_PROCOPID;
				a_PATH[index] = i_PATH;
				if (++index >= ArraySize) { cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			}
			/// <summary>
			/// Reads a set of rows from TB_B_VOLTKS_SBPATHS and retrieves them into a new TB_B_VOLTKS_SBPATHS object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the TB_B_VOLTKS_SBPATHS class that can be used to read the retrieved data.</returns>
			static public TB_B_VOLTKS_SBPATHS SelectWhere(string wherestr, string orderstr)
			{
				TB_B_VOLTKS_SBPATHS newobj = new TB_B_VOLTKS_SBPATHS();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_EVENTBRICK,ID_RECONSTRUCTION,ID_OPTION,ID_VOLUMETRACK,EXTENT,ID_SCANBACK_PROCOPID,PATH FROM TB_B_VOLTKS_SBPATHS" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO TB_B_VOLTKS_SBPATHS (ID_EVENTBRICK,ID_RECONSTRUCTION,ID_OPTION,ID_VOLUMETRACK,EXTENT,ID_SCANBACK_PROCOPID,PATH) VALUES (:p_1,:p_2,:p_3,:p_4,:p_5,:p_6,:p_7)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_EVENTBRICK;
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_RECONSTRUCTION;
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_OPTION;
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_VOLUMETRACK;
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input).Value = a_EXTENT;
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_SCANBACK_PROCOPID;
				newcmd.Parameters.Add("p_7", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_PATH;
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the TB_CS_CANDIDATES table in the DB.
		/// For data insertion, the Insert method is used. Rows are inserted one by one.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class TB_CS_CANDIDATES
		{
			internal TB_CS_CANDIDATES() {}
			System.Data.DataRowCollection m_DRC;
			/// <summary>
			/// Retrieves ID_EVENTBRICK for the current row.
			/// </summary>
			public long _ID_EVENTBRICK
			{
				get
				{
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			/// <summary>
			/// Retrieves ID_PROCESSOPERATION for the current row.
			/// </summary>
			public long _ID_PROCESSOPERATION
			{
				get
				{
					return System.Convert.ToInt64(m_DR[1]);
				}
			}
			/// <summary>
			/// Retrieves ID_EVENT for the current row.
			/// </summary>
			public long _ID_EVENT
			{
				get
				{
					return System.Convert.ToInt64(m_DR[2]);
				}
			}
			/// <summary>
			/// Retrieves CANDIDATE for the current row.
			/// </summary>
			public long _CANDIDATE
			{
				get
				{
					return System.Convert.ToInt64(m_DR[3]);
				}
			}
			/// <summary>
			/// Retrieves ID for the current row.
			/// </summary>
			public long _ID
			{
				get
				{
					return System.Convert.ToInt64(m_DR[4]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is inserted immediately.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">the value to be inserted for ID_EVENTBRICK.</param>
			/// <param name="i_ID_PROCESSOPERATION">the value to be inserted for ID_PROCESSOPERATION.</param>
			/// <param name="i_ID_EVENT">the value to be inserted for ID_EVENT.</param>
			/// <param name="i_CANDIDATE">the value to be inserted for CANDIDATE.</param>
			/// <param name="i_ID">the value to be inserted for ID. This value is actually used only if this method call is involved in data publication/replication, otherwise the actual value is generated by the OPERA DB and the supplied value is ignored.</param>
			/// <returns>the value of ID for the new row.</returns>
			static public long Insert(long i_ID_EVENTBRICK,long i_ID_PROCESSOPERATION,long i_ID_EVENT,long i_CANDIDATE,long i_ID)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = i_ID_EVENTBRICK;
				cmd.Parameters[1].Value = i_ID_PROCESSOPERATION;
				cmd.Parameters[2].Value = i_ID_EVENT;
				cmd.Parameters[3].Value = i_CANDIDATE;
				cmd.Parameters[4].Value = i_ID;
				cmd.ExecuteNonQuery();
				return SySal.OperaDb.Convert.ToInt64(cmd.Parameters[5].Value);
			}
			/// <summary>
			/// Reads a set of rows from TB_CS_CANDIDATES and retrieves them into a new TB_CS_CANDIDATES object.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_PROCESSOPERATION">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_EVENT">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_CANDIDATE">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="order">the ordering scheme to be applied to returned rows. This applies to all columns in the primary key.</param>
			/// <returns>a new instance of the TB_CS_CANDIDATES class that can be used to read the retrieved data.</returns>
			static public TB_CS_CANDIDATES SelectPrimaryKey(object i_ID_EVENTBRICK,object i_ID_PROCESSOPERATION,object i_ID_EVENT,object i_CANDIDATE, OrderBy order)
			{
				string wherestr = "";
				string wtempstr = "";
				if (i_ID_EVENTBRICK != null)
				{
					if (i_ID_EVENTBRICK == System.DBNull.Value) wtempstr = "ID_EVENTBRICK IS NULL";
					else wtempstr = "ID_EVENTBRICK = " + i_ID_EVENTBRICK.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_PROCESSOPERATION != null)
				{
					if (i_ID_PROCESSOPERATION == System.DBNull.Value) wtempstr = "ID_PROCESSOPERATION IS NULL";
					else wtempstr = "ID_PROCESSOPERATION = " + i_ID_PROCESSOPERATION.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_EVENT != null)
				{
					if (i_ID_EVENT == System.DBNull.Value) wtempstr = "ID_EVENT IS NULL";
					else wtempstr = "ID_EVENT = " + i_ID_EVENT.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_CANDIDATE != null)
				{
					if (i_CANDIDATE == System.DBNull.Value) wtempstr = "CANDIDATE IS NULL";
					else wtempstr = "CANDIDATE = " + i_CANDIDATE.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (order == OrderBy.Ascending) return SelectWhere(wherestr, "ID_EVENTBRICK ASC,ID_PROCESSOPERATION ASC,ID_EVENT ASC,CANDIDATE ASC");
				else if (order == OrderBy.Descending) return SelectWhere(wherestr, "ID_EVENTBRICK DESC,ID_PROCESSOPERATION DESC,ID_EVENT DESC,CANDIDATE DESC");
				return SelectWhere(wherestr, null);
			}
			/// <summary>
			/// Reads a set of rows from TB_CS_CANDIDATES and retrieves them into a new TB_CS_CANDIDATES object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the TB_CS_CANDIDATES class that can be used to read the retrieved data.</returns>
			static public TB_CS_CANDIDATES SelectWhere(string wherestr, string orderstr)
			{
				TB_CS_CANDIDATES newobj = new TB_CS_CANDIDATES();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_EVENTBRICK,ID_PROCESSOPERATION,ID_EVENT,CANDIDATE,ID FROM TB_CS_CANDIDATES" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO TB_CS_CANDIDATES (ID_EVENTBRICK,ID_PROCESSOPERATION,ID_EVENT,CANDIDATE,ID) VALUES (:p_1,:p_2,:p_3,:p_4,:p_5) RETURNING ID INTO :o_5");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("o_5", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Output);
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the TB_CS_CANDIDATE_CHECKS table in the DB.
		/// For data insertion, the Insert and Flush static methods are used. This class inserts data in batches.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class TB_CS_CANDIDATE_CHECKS
		{
			internal TB_CS_CANDIDATE_CHECKS() {}
			System.Data.DataRowCollection m_DRC;
			const int ArraySize = 100;
			static int index = 0;
			/// <summary>
			/// Since this class uses batch insertion, this method must be called to ensure that all rows are actually inserted into TB_CS_CANDIDATE_CHECKS. Failure to do so will result in incomplete writes.
			/// </summary>
			static public void Flush() { if (cmd == null || index == 0) return; cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			private static long [] a_ID_EVENTBRICK = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_EVENTBRICK for the current row.
			/// </summary>
			public long _ID_EVENTBRICK
			{
				get
				{
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			private static long [] a_ID_CANDIDATE = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_CANDIDATE for the current row.
			/// </summary>
			public long _ID_CANDIDATE
			{
				get
				{
					return System.Convert.ToInt64(m_DR[1]);
				}
			}
			private static long [] a_ID_PROCESSOPERATION = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_PROCESSOPERATION for the current row.
			/// </summary>
			public long _ID_PROCESSOPERATION
			{
				get
				{
					return System.Convert.ToInt64(m_DR[2]);
				}
			}
			private static long [] a_ID_ZONE = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_ZONE for the current row.
			/// </summary>
			public long _ID_ZONE
			{
				get
				{
					return System.Convert.ToInt64(m_DR[3]);
				}
			}
			private static long [] a_SIDE = new long[ArraySize];
			/// <summary>
			/// Retrieves SIDE for the current row.
			/// </summary>
			public long _SIDE
			{
				get
				{
					return System.Convert.ToInt64(m_DR[4]);
				}
			}
			private static object [] a_GRAINS = new object[ArraySize];
			/// <summary>
			/// Retrieves GRAINS for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _GRAINS
			{
				get
				{
					if (m_DR[5] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[5]);
				}
			}
			private static object [] a_POSX = new object[ArraySize];
			/// <summary>
			/// Retrieves POSX for the current row. The return value can be System.DBNull.Value or a value that can be cast to double.
			/// </summary>
			public object _POSX
			{
				get
				{
					if (m_DR[6] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDouble(m_DR[6]);
				}
			}
			private static object [] a_POSY = new object[ArraySize];
			/// <summary>
			/// Retrieves POSY for the current row. The return value can be System.DBNull.Value or a value that can be cast to double.
			/// </summary>
			public object _POSY
			{
				get
				{
					if (m_DR[7] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDouble(m_DR[7]);
				}
			}
			private static object [] a_SLOPEX = new object[ArraySize];
			/// <summary>
			/// Retrieves SLOPEX for the current row. The return value can be System.DBNull.Value or a value that can be cast to double.
			/// </summary>
			public object _SLOPEX
			{
				get
				{
					if (m_DR[8] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDouble(m_DR[8]);
				}
			}
			private static object [] a_SLOPEY = new object[ArraySize];
			/// <summary>
			/// Retrieves SLOPEY for the current row. The return value can be System.DBNull.Value or a value that can be cast to double.
			/// </summary>
			public object _SLOPEY
			{
				get
				{
					if (m_DR[9] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDouble(m_DR[9]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is actually inserted later, in a batch insertion command. The Flush method must be called to ensure all rows are actually written.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">the value to be inserted for ID_EVENTBRICK.</param>
			/// <param name="i_ID_CANDIDATE">the value to be inserted for ID_CANDIDATE.</param>
			/// <param name="i_ID_PROCESSOPERATION">the value to be inserted for ID_PROCESSOPERATION.</param>
			/// <param name="i_ID_ZONE">the value to be inserted for ID_ZONE.</param>
			/// <param name="i_SIDE">the value to be inserted for SIDE.</param>
			/// <param name="i_GRAINS">the value to be inserted for GRAINS. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_POSX">the value to be inserted for POSX. The value for this parameter can be double or System.DBNull.Value.</param>
			/// <param name="i_POSY">the value to be inserted for POSY. The value for this parameter can be double or System.DBNull.Value.</param>
			/// <param name="i_SLOPEX">the value to be inserted for SLOPEX. The value for this parameter can be double or System.DBNull.Value.</param>
			/// <param name="i_SLOPEY">the value to be inserted for SLOPEY. The value for this parameter can be double or System.DBNull.Value.</param>
			static public void Insert(long i_ID_EVENTBRICK,long i_ID_CANDIDATE,long i_ID_PROCESSOPERATION,long i_ID_ZONE,long i_SIDE,object i_GRAINS,object i_POSX,object i_POSY,object i_SLOPEX,object i_SLOPEY)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				a_ID_EVENTBRICK[index] = i_ID_EVENTBRICK;
				a_ID_CANDIDATE[index] = i_ID_CANDIDATE;
				a_ID_PROCESSOPERATION[index] = i_ID_PROCESSOPERATION;
				a_ID_ZONE[index] = i_ID_ZONE;
				a_SIDE[index] = i_SIDE;
				a_GRAINS[index] = i_GRAINS;
				a_POSX[index] = i_POSX;
				a_POSY[index] = i_POSY;
				a_SLOPEX[index] = i_SLOPEX;
				a_SLOPEY[index] = i_SLOPEY;
				if (++index >= ArraySize) { cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			}
			/// <summary>
			/// Reads a set of rows from TB_CS_CANDIDATE_CHECKS and retrieves them into a new TB_CS_CANDIDATE_CHECKS object.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_CANDIDATE">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_PROCESSOPERATION">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_ZONE">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_SIDE">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="order">the ordering scheme to be applied to returned rows. This applies to all columns in the primary key.</param>
			/// <returns>a new instance of the TB_CS_CANDIDATE_CHECKS class that can be used to read the retrieved data.</returns>
			static public TB_CS_CANDIDATE_CHECKS SelectPrimaryKey(object i_ID_EVENTBRICK,object i_ID_CANDIDATE,object i_ID_PROCESSOPERATION,object i_ID_ZONE,object i_SIDE, OrderBy order)
			{
				string wherestr = "";
				string wtempstr = "";
				if (i_ID_EVENTBRICK != null)
				{
					if (i_ID_EVENTBRICK == System.DBNull.Value) wtempstr = "ID_EVENTBRICK IS NULL";
					else wtempstr = "ID_EVENTBRICK = " + i_ID_EVENTBRICK.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_CANDIDATE != null)
				{
					if (i_ID_CANDIDATE == System.DBNull.Value) wtempstr = "ID_CANDIDATE IS NULL";
					else wtempstr = "ID_CANDIDATE = " + i_ID_CANDIDATE.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_PROCESSOPERATION != null)
				{
					if (i_ID_PROCESSOPERATION == System.DBNull.Value) wtempstr = "ID_PROCESSOPERATION IS NULL";
					else wtempstr = "ID_PROCESSOPERATION = " + i_ID_PROCESSOPERATION.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_ZONE != null)
				{
					if (i_ID_ZONE == System.DBNull.Value) wtempstr = "ID_ZONE IS NULL";
					else wtempstr = "ID_ZONE = " + i_ID_ZONE.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_SIDE != null)
				{
					if (i_SIDE == System.DBNull.Value) wtempstr = "SIDE IS NULL";
					else wtempstr = "SIDE = " + i_SIDE.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (order == OrderBy.Ascending) return SelectWhere(wherestr, "ID_EVENTBRICK ASC,ID_CANDIDATE ASC,ID_PROCESSOPERATION ASC,ID_ZONE ASC,SIDE ASC");
				else if (order == OrderBy.Descending) return SelectWhere(wherestr, "ID_EVENTBRICK DESC,ID_CANDIDATE DESC,ID_PROCESSOPERATION DESC,ID_ZONE DESC,SIDE DESC");
				return SelectWhere(wherestr, null);
			}
			/// <summary>
			/// Reads a set of rows from TB_CS_CANDIDATE_CHECKS and retrieves them into a new TB_CS_CANDIDATE_CHECKS object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the TB_CS_CANDIDATE_CHECKS class that can be used to read the retrieved data.</returns>
			static public TB_CS_CANDIDATE_CHECKS SelectWhere(string wherestr, string orderstr)
			{
				TB_CS_CANDIDATE_CHECKS newobj = new TB_CS_CANDIDATE_CHECKS();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_EVENTBRICK,ID_CANDIDATE,ID_PROCESSOPERATION,ID_ZONE,SIDE,GRAINS,POSX,POSY,SLOPEX,SLOPEY FROM TB_CS_CANDIDATE_CHECKS" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO TB_CS_CANDIDATE_CHECKS (ID_EVENTBRICK,ID_CANDIDATE,ID_PROCESSOPERATION,ID_ZONE,SIDE,GRAINS,POSX,POSY,SLOPEX,SLOPEY) VALUES (:p_1,:p_2,:p_3,:p_4,:p_5,:p_6,:p_7,:p_8,:p_9,:p_10)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_EVENTBRICK;
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_CANDIDATE;
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_PROCESSOPERATION;
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_ZONE;
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_SIDE;
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_GRAINS;
				newcmd.Parameters.Add("p_7", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_POSX;
				newcmd.Parameters.Add("p_8", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_POSY;
				newcmd.Parameters.Add("p_9", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_SLOPEX;
				newcmd.Parameters.Add("p_10", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_SLOPEY;
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the TB_CS_CANDIDATE_TRACKS table in the DB.
		/// For data insertion, the Insert and Flush static methods are used. This class inserts data in batches.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class TB_CS_CANDIDATE_TRACKS
		{
			internal TB_CS_CANDIDATE_TRACKS() {}
			System.Data.DataRowCollection m_DRC;
			const int ArraySize = 100;
			static int index = 0;
			/// <summary>
			/// Since this class uses batch insertion, this method must be called to ensure that all rows are actually inserted into TB_CS_CANDIDATE_TRACKS. Failure to do so will result in incomplete writes.
			/// </summary>
			static public void Flush() { if (cmd == null || index == 0) return; cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			private static long [] a_ID_EVENTBRICK = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_EVENTBRICK for the current row.
			/// </summary>
			public long _ID_EVENTBRICK
			{
				get
				{
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			private static long [] a_ID_CANDIDATE = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_CANDIDATE for the current row.
			/// </summary>
			public long _ID_CANDIDATE
			{
				get
				{
					return System.Convert.ToInt64(m_DR[1]);
				}
			}
			private static long [] a_ID_ZONE = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_ZONE for the current row.
			/// </summary>
			public long _ID_ZONE
			{
				get
				{
					return System.Convert.ToInt64(m_DR[2]);
				}
			}
			private static int [] a_SIDE = new int[ArraySize];
			/// <summary>
			/// Retrieves SIDE for the current row.
			/// </summary>
			public int _SIDE
			{
				get
				{
					return System.Convert.ToInt32(m_DR[3]);
				}
			}
			private static long [] a_ID_MICROTRACK = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_MICROTRACK for the current row.
			/// </summary>
			public long _ID_MICROTRACK
			{
				get
				{
					return System.Convert.ToInt64(m_DR[4]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is actually inserted later, in a batch insertion command. The Flush method must be called to ensure all rows are actually written.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">the value to be inserted for ID_EVENTBRICK.</param>
			/// <param name="i_ID_CANDIDATE">the value to be inserted for ID_CANDIDATE.</param>
			/// <param name="i_ID_ZONE">the value to be inserted for ID_ZONE.</param>
			/// <param name="i_SIDE">the value to be inserted for SIDE.</param>
			/// <param name="i_ID_MICROTRACK">the value to be inserted for ID_MICROTRACK.</param>
			static public void Insert(long i_ID_EVENTBRICK,long i_ID_CANDIDATE,long i_ID_ZONE,int i_SIDE,long i_ID_MICROTRACK)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				a_ID_EVENTBRICK[index] = i_ID_EVENTBRICK;
				a_ID_CANDIDATE[index] = i_ID_CANDIDATE;
				a_ID_ZONE[index] = i_ID_ZONE;
				a_SIDE[index] = i_SIDE;
				a_ID_MICROTRACK[index] = i_ID_MICROTRACK;
				if (++index >= ArraySize) { cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			}
			/// <summary>
			/// Reads a set of rows from TB_CS_CANDIDATE_TRACKS and retrieves them into a new TB_CS_CANDIDATE_TRACKS object.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_CANDIDATE">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_ZONE">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_SIDE">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="order">the ordering scheme to be applied to returned rows. This applies to all columns in the primary key.</param>
			/// <returns>a new instance of the TB_CS_CANDIDATE_TRACKS class that can be used to read the retrieved data.</returns>
			static public TB_CS_CANDIDATE_TRACKS SelectPrimaryKey(object i_ID_EVENTBRICK,object i_ID_CANDIDATE,object i_ID_ZONE,object i_SIDE, OrderBy order)
			{
				string wherestr = "";
				string wtempstr = "";
				if (i_ID_EVENTBRICK != null)
				{
					if (i_ID_EVENTBRICK == System.DBNull.Value) wtempstr = "ID_EVENTBRICK IS NULL";
					else wtempstr = "ID_EVENTBRICK = " + i_ID_EVENTBRICK.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_CANDIDATE != null)
				{
					if (i_ID_CANDIDATE == System.DBNull.Value) wtempstr = "ID_CANDIDATE IS NULL";
					else wtempstr = "ID_CANDIDATE = " + i_ID_CANDIDATE.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_ZONE != null)
				{
					if (i_ID_ZONE == System.DBNull.Value) wtempstr = "ID_ZONE IS NULL";
					else wtempstr = "ID_ZONE = " + i_ID_ZONE.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_SIDE != null)
				{
					if (i_SIDE == System.DBNull.Value) wtempstr = "SIDE IS NULL";
					else wtempstr = "SIDE = " + i_SIDE.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (order == OrderBy.Ascending) return SelectWhere(wherestr, "ID_EVENTBRICK ASC,ID_CANDIDATE ASC,ID_ZONE ASC,SIDE ASC");
				else if (order == OrderBy.Descending) return SelectWhere(wherestr, "ID_EVENTBRICK DESC,ID_CANDIDATE DESC,ID_ZONE DESC,SIDE DESC");
				return SelectWhere(wherestr, null);
			}
			/// <summary>
			/// Reads a set of rows from TB_CS_CANDIDATE_TRACKS and retrieves them into a new TB_CS_CANDIDATE_TRACKS object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the TB_CS_CANDIDATE_TRACKS class that can be used to read the retrieved data.</returns>
			static public TB_CS_CANDIDATE_TRACKS SelectWhere(string wherestr, string orderstr)
			{
				TB_CS_CANDIDATE_TRACKS newobj = new TB_CS_CANDIDATE_TRACKS();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_EVENTBRICK,ID_CANDIDATE,ID_ZONE,SIDE,ID_MICROTRACK FROM TB_CS_CANDIDATE_TRACKS" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO TB_CS_CANDIDATE_TRACKS (ID_EVENTBRICK,ID_CANDIDATE,ID_ZONE,SIDE,ID_MICROTRACK) VALUES (:p_1,:p_2,:p_3,:p_4,:p_5)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_EVENTBRICK;
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_CANDIDATE;
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_ZONE;
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Int, System.Data.ParameterDirection.Input).Value = a_SIDE;
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_MICROTRACK;
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the TB_EVENTBRICKS table in the DB.
		/// For data insertion, the Insert and Flush static methods are used. This class inserts data in batches.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class TB_EVENTBRICKS
		{
			internal TB_EVENTBRICKS() {}
			System.Data.DataRowCollection m_DRC;
			const int ArraySize = 100;
			static int index = 0;
			/// <summary>
			/// Since this class uses batch insertion, this method must be called to ensure that all rows are actually inserted into TB_EVENTBRICKS. Failure to do so will result in incomplete writes.
			/// </summary>
			static public void Flush() { if (cmd == null || index == 0) return; cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			private static long [] a_ID = new long[ArraySize];
			/// <summary>
			/// Retrieves ID for the current row.
			/// </summary>
			public long _ID
			{
				get
				{
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			private static double [] a_MINX = new double[ArraySize];
			/// <summary>
			/// Retrieves MINX for the current row.
			/// </summary>
			public double _MINX
			{
				get
				{
					return System.Convert.ToDouble(m_DR[1]);
				}
			}
			private static double [] a_MAXX = new double[ArraySize];
			/// <summary>
			/// Retrieves MAXX for the current row.
			/// </summary>
			public double _MAXX
			{
				get
				{
					return System.Convert.ToDouble(m_DR[2]);
				}
			}
			private static double [] a_MINY = new double[ArraySize];
			/// <summary>
			/// Retrieves MINY for the current row.
			/// </summary>
			public double _MINY
			{
				get
				{
					return System.Convert.ToDouble(m_DR[3]);
				}
			}
			private static double [] a_MAXY = new double[ArraySize];
			/// <summary>
			/// Retrieves MAXY for the current row.
			/// </summary>
			public double _MAXY
			{
				get
				{
					return System.Convert.ToDouble(m_DR[4]);
				}
			}
			private static double [] a_MINZ = new double[ArraySize];
			/// <summary>
			/// Retrieves MINZ for the current row.
			/// </summary>
			public double _MINZ
			{
				get
				{
					return System.Convert.ToDouble(m_DR[5]);
				}
			}
			private static double [] a_MAXZ = new double[ArraySize];
			/// <summary>
			/// Retrieves MAXZ for the current row.
			/// </summary>
			public double _MAXZ
			{
				get
				{
					return System.Convert.ToDouble(m_DR[6]);
				}
			}
			private static string [] a_ID_SET = new string[ArraySize];
			/// <summary>
			/// Retrieves ID_SET for the current row.
			/// </summary>
			public string _ID_SET
			{
				get
				{
					return System.Convert.ToString(m_DR[7]);
				}
			}
			private static long [] a_ID_BRICK = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_BRICK for the current row.
			/// </summary>
			public long _ID_BRICK
			{
				get
				{
					return System.Convert.ToInt64(m_DR[8]);
				}
			}
			private static object [] a_ZEROX = new object[ArraySize];
			/// <summary>
			/// Retrieves ZEROX for the current row. The return value can be System.DBNull.Value or a value that can be cast to double.
			/// </summary>
			public object _ZEROX
			{
				get
				{
					if (m_DR[9] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDouble(m_DR[9]);
				}
			}
			private static object [] a_ZEROY = new object[ArraySize];
			/// <summary>
			/// Retrieves ZEROY for the current row. The return value can be System.DBNull.Value or a value that can be cast to double.
			/// </summary>
			public object _ZEROY
			{
				get
				{
					if (m_DR[10] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDouble(m_DR[10]);
				}
			}
			private static object [] a_ZEROZ = new object[ArraySize];
			/// <summary>
			/// Retrieves ZEROZ for the current row. The return value can be System.DBNull.Value or a value that can be cast to double.
			/// </summary>
			public object _ZEROZ
			{
				get
				{
					if (m_DR[11] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDouble(m_DR[11]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is actually inserted later, in a batch insertion command. The Flush method must be called to ensure all rows are actually written.
			/// </summary>
			/// <param name="i_ID">the value to be inserted for ID.</param>
			/// <param name="i_MINX">the value to be inserted for MINX.</param>
			/// <param name="i_MAXX">the value to be inserted for MAXX.</param>
			/// <param name="i_MINY">the value to be inserted for MINY.</param>
			/// <param name="i_MAXY">the value to be inserted for MAXY.</param>
			/// <param name="i_MINZ">the value to be inserted for MINZ.</param>
			/// <param name="i_MAXZ">the value to be inserted for MAXZ.</param>
			/// <param name="i_ID_SET">the value to be inserted for ID_SET.</param>
			/// <param name="i_ID_BRICK">the value to be inserted for ID_BRICK.</param>
			/// <param name="i_ZEROX">the value to be inserted for ZEROX. The value for this parameter can be double or System.DBNull.Value.</param>
			/// <param name="i_ZEROY">the value to be inserted for ZEROY. The value for this parameter can be double or System.DBNull.Value.</param>
			/// <param name="i_ZEROZ">the value to be inserted for ZEROZ. The value for this parameter can be double or System.DBNull.Value.</param>
			static public void Insert(long i_ID,double i_MINX,double i_MAXX,double i_MINY,double i_MAXY,double i_MINZ,double i_MAXZ,string i_ID_SET,long i_ID_BRICK,object i_ZEROX,object i_ZEROY,object i_ZEROZ)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				a_ID[index] = i_ID;
				a_MINX[index] = i_MINX;
				a_MAXX[index] = i_MAXX;
				a_MINY[index] = i_MINY;
				a_MAXY[index] = i_MAXY;
				a_MINZ[index] = i_MINZ;
				a_MAXZ[index] = i_MAXZ;
				a_ID_SET[index] = i_ID_SET;
				a_ID_BRICK[index] = i_ID_BRICK;
				a_ZEROX[index] = i_ZEROX;
				a_ZEROY[index] = i_ZEROY;
				a_ZEROZ[index] = i_ZEROZ;
				if (++index >= ArraySize) { cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			}
			/// <summary>
			/// Reads a set of rows from TB_EVENTBRICKS and retrieves them into a new TB_EVENTBRICKS object.
			/// </summary>
			/// <param name="i_ID">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="order">the ordering scheme to be applied to returned rows.</param>
			/// <returns>a new instance of the TB_EVENTBRICKS class that can be used to read the retrieved data.</returns>
			static public TB_EVENTBRICKS SelectPrimaryKey(object i_ID, OrderBy order)
			{
				string wherestr = "";
				string wtempstr = "";
				if (i_ID != null)
				{
					if (i_ID == System.DBNull.Value) wtempstr = "ID IS NULL";
					else wtempstr = "ID = " + i_ID.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (order == OrderBy.Ascending) return SelectWhere(wherestr, "ID ASC");
				else if (order == OrderBy.Descending) return SelectWhere(wherestr, "ID DESC");
				return SelectWhere(wherestr, null);
			}
			/// <summary>
			/// Reads a set of rows from TB_EVENTBRICKS and retrieves them into a new TB_EVENTBRICKS object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the TB_EVENTBRICKS class that can be used to read the retrieved data.</returns>
			static public TB_EVENTBRICKS SelectWhere(string wherestr, string orderstr)
			{
				TB_EVENTBRICKS newobj = new TB_EVENTBRICKS();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID,MINX,MAXX,MINY,MAXY,MINZ,MAXZ,ID_SET,ID_BRICK,ZEROX,ZEROY,ZEROZ FROM TB_EVENTBRICKS" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO TB_EVENTBRICKS (ID,MINX,MAXX,MINY,MAXY,MINZ,MAXZ,ID_SET,ID_BRICK,ZEROX,ZEROY,ZEROZ) VALUES (:p_1,:p_2,:p_3,:p_4,:p_5,:p_6,:p_7,:p_8,:p_9,:p_10,:p_11,:p_12)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID;
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_MINX;
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_MAXX;
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_MINY;
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_MAXY;
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_MINZ;
				newcmd.Parameters.Add("p_7", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_MAXZ;
				newcmd.Parameters.Add("p_8", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input).Value = a_ID_SET;
				newcmd.Parameters.Add("p_9", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_BRICK;
				newcmd.Parameters.Add("p_10", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_ZEROX;
				newcmd.Parameters.Add("p_11", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_ZEROY;
				newcmd.Parameters.Add("p_12", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_ZEROZ;
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the TB_GRAINS table in the DB.
		/// For data insertion, the Insert and Flush static methods are used. This class inserts data in batches.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class TB_GRAINS
		{
			internal TB_GRAINS() {}
			System.Data.DataRowCollection m_DRC;
			const int ArraySize = 100;
			static int index = 0;
			/// <summary>
			/// Since this class uses batch insertion, this method must be called to ensure that all rows are actually inserted into TB_GRAINS. Failure to do so will result in incomplete writes.
			/// </summary>
			static public void Flush() { if (cmd == null || index == 0) return; cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			private static long [] a_ID_EVENTBRICK = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_EVENTBRICK for the current row.
			/// </summary>
			public long _ID_EVENTBRICK
			{
				get
				{
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			private static long [] a_ID_ZONE = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_ZONE for the current row.
			/// </summary>
			public long _ID_ZONE
			{
				get
				{
					return System.Convert.ToInt64(m_DR[1]);
				}
			}
			private static int [] a_SIDE = new int[ArraySize];
			/// <summary>
			/// Retrieves SIDE for the current row.
			/// </summary>
			public int _SIDE
			{
				get
				{
					return System.Convert.ToInt32(m_DR[2]);
				}
			}
			private static long [] a_ID_MIPMICROTRACK = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_MIPMICROTRACK for the current row.
			/// </summary>
			public long _ID_MIPMICROTRACK
			{
				get
				{
					return System.Convert.ToInt64(m_DR[3]);
				}
			}
			private static long [] a_ID = new long[ArraySize];
			/// <summary>
			/// Retrieves ID for the current row.
			/// </summary>
			public long _ID
			{
				get
				{
					return System.Convert.ToInt64(m_DR[4]);
				}
			}
			private static double [] a_X = new double[ArraySize];
			/// <summary>
			/// Retrieves X for the current row.
			/// </summary>
			public double _X
			{
				get
				{
					return System.Convert.ToDouble(m_DR[5]);
				}
			}
			private static double [] a_Y = new double[ArraySize];
			/// <summary>
			/// Retrieves Y for the current row.
			/// </summary>
			public double _Y
			{
				get
				{
					return System.Convert.ToDouble(m_DR[6]);
				}
			}
			private static double [] a_Z = new double[ArraySize];
			/// <summary>
			/// Retrieves Z for the current row.
			/// </summary>
			public double _Z
			{
				get
				{
					return System.Convert.ToDouble(m_DR[7]);
				}
			}
			private static object [] a_AREA = new object[ArraySize];
			/// <summary>
			/// Retrieves AREA for the current row. The return value can be System.DBNull.Value or a value that can be cast to int.
			/// </summary>
			public object _AREA
			{
				get
				{
					if (m_DR[8] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt32(m_DR[8]);
				}
			}
			private static object [] a_DARKNESS = new object[ArraySize];
			/// <summary>
			/// Retrieves DARKNESS for the current row. The return value can be System.DBNull.Value or a value that can be cast to double.
			/// </summary>
			public object _DARKNESS
			{
				get
				{
					if (m_DR[9] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDouble(m_DR[9]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is actually inserted later, in a batch insertion command. The Flush method must be called to ensure all rows are actually written.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">the value to be inserted for ID_EVENTBRICK.</param>
			/// <param name="i_ID_ZONE">the value to be inserted for ID_ZONE.</param>
			/// <param name="i_SIDE">the value to be inserted for SIDE.</param>
			/// <param name="i_ID_MIPMICROTRACK">the value to be inserted for ID_MIPMICROTRACK.</param>
			/// <param name="i_ID">the value to be inserted for ID.</param>
			/// <param name="i_X">the value to be inserted for X.</param>
			/// <param name="i_Y">the value to be inserted for Y.</param>
			/// <param name="i_Z">the value to be inserted for Z.</param>
			/// <param name="i_AREA">the value to be inserted for AREA. The value for this parameter can be int or System.DBNull.Value.</param>
			/// <param name="i_DARKNESS">the value to be inserted for DARKNESS. The value for this parameter can be double or System.DBNull.Value.</param>
			static public void Insert(long i_ID_EVENTBRICK,long i_ID_ZONE,int i_SIDE,long i_ID_MIPMICROTRACK,long i_ID,double i_X,double i_Y,double i_Z,object i_AREA,object i_DARKNESS)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				a_ID_EVENTBRICK[index] = i_ID_EVENTBRICK;
				a_ID_ZONE[index] = i_ID_ZONE;
				a_SIDE[index] = i_SIDE;
				a_ID_MIPMICROTRACK[index] = i_ID_MIPMICROTRACK;
				a_ID[index] = i_ID;
				a_X[index] = i_X;
				a_Y[index] = i_Y;
				a_Z[index] = i_Z;
				a_AREA[index] = i_AREA;
				a_DARKNESS[index] = i_DARKNESS;
				if (++index >= ArraySize) { cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			}
			/// <summary>
			/// Reads a set of rows from TB_GRAINS and retrieves them into a new TB_GRAINS object.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_ZONE">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_SIDE">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_MIPMICROTRACK">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="order">the ordering scheme to be applied to returned rows. This applies to all columns in the primary key.</param>
			/// <returns>a new instance of the TB_GRAINS class that can be used to read the retrieved data.</returns>
			static public TB_GRAINS SelectPrimaryKey(object i_ID_EVENTBRICK,object i_ID_ZONE,object i_SIDE,object i_ID_MIPMICROTRACK,object i_ID, OrderBy order)
			{
				string wherestr = "";
				string wtempstr = "";
				if (i_ID_EVENTBRICK != null)
				{
					if (i_ID_EVENTBRICK == System.DBNull.Value) wtempstr = "ID_EVENTBRICK IS NULL";
					else wtempstr = "ID_EVENTBRICK = " + i_ID_EVENTBRICK.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_ZONE != null)
				{
					if (i_ID_ZONE == System.DBNull.Value) wtempstr = "ID_ZONE IS NULL";
					else wtempstr = "ID_ZONE = " + i_ID_ZONE.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_SIDE != null)
				{
					if (i_SIDE == System.DBNull.Value) wtempstr = "SIDE IS NULL";
					else wtempstr = "SIDE = " + i_SIDE.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_MIPMICROTRACK != null)
				{
					if (i_ID_MIPMICROTRACK == System.DBNull.Value) wtempstr = "ID_MIPMICROTRACK IS NULL";
					else wtempstr = "ID_MIPMICROTRACK = " + i_ID_MIPMICROTRACK.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID != null)
				{
					if (i_ID == System.DBNull.Value) wtempstr = "ID IS NULL";
					else wtempstr = "ID = " + i_ID.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (order == OrderBy.Ascending) return SelectWhere(wherestr, "ID_EVENTBRICK ASC,ID_ZONE ASC,SIDE ASC,ID_MIPMICROTRACK ASC,ID ASC");
				else if (order == OrderBy.Descending) return SelectWhere(wherestr, "ID_EVENTBRICK DESC,ID_ZONE DESC,SIDE DESC,ID_MIPMICROTRACK DESC,ID DESC");
				return SelectWhere(wherestr, null);
			}
			/// <summary>
			/// Reads a set of rows from TB_GRAINS and retrieves them into a new TB_GRAINS object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the TB_GRAINS class that can be used to read the retrieved data.</returns>
			static public TB_GRAINS SelectWhere(string wherestr, string orderstr)
			{
				TB_GRAINS newobj = new TB_GRAINS();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_EVENTBRICK,ID_ZONE,SIDE,ID_MIPMICROTRACK,ID,X,Y,Z,AREA,DARKNESS FROM TB_GRAINS" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO TB_GRAINS (ID_EVENTBRICK,ID_ZONE,SIDE,ID_MIPMICROTRACK,ID,X,Y,Z,AREA,DARKNESS) VALUES (:p_1,:p_2,:p_3,:p_4,:p_5,:p_6,:p_7,:p_8,:p_9,:p_10)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_EVENTBRICK;
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_ZONE;
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Int, System.Data.ParameterDirection.Input).Value = a_SIDE;
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_MIPMICROTRACK;
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID;
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_X;
				newcmd.Parameters.Add("p_7", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_Y;
				newcmd.Parameters.Add("p_8", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_Z;
				newcmd.Parameters.Add("p_9", SySal.OperaDb.OperaDbType.Int, System.Data.ParameterDirection.Input).Value = a_AREA;
				newcmd.Parameters.Add("p_10", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_DARKNESS;
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the TB_MACHINES table in the DB.
		/// For data insertion, the Insert method is used. Rows are inserted one by one.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class TB_MACHINES
		{
			internal TB_MACHINES() {}
			System.Data.DataRowCollection m_DRC;
			/// <summary>
			/// Retrieves ID for the current row.
			/// </summary>
			public long _ID
			{
				get
				{
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			/// <summary>
			/// Retrieves ID_SITE for the current row.
			/// </summary>
			public long _ID_SITE
			{
				get
				{
					return System.Convert.ToInt64(m_DR[1]);
				}
			}
			/// <summary>
			/// Retrieves NAME for the current row.
			/// </summary>
			public string _NAME
			{
				get
				{
					return System.Convert.ToString(m_DR[2]);
				}
			}
			/// <summary>
			/// Retrieves ADDRESS for the current row.
			/// </summary>
			public string _ADDRESS
			{
				get
				{
					return System.Convert.ToString(m_DR[3]);
				}
			}
			/// <summary>
			/// Retrieves ISSCANNINGSERVER for the current row.
			/// </summary>
			public int _ISSCANNINGSERVER
			{
				get
				{
					return System.Convert.ToInt32(m_DR[4]);
				}
			}
			/// <summary>
			/// Retrieves ISBATCHSERVER for the current row.
			/// </summary>
			public int _ISBATCHSERVER
			{
				get
				{
					return System.Convert.ToInt32(m_DR[5]);
				}
			}
			/// <summary>
			/// Retrieves ISDATAPROCESSINGSERVER for the current row.
			/// </summary>
			public int _ISDATAPROCESSINGSERVER
			{
				get
				{
					return System.Convert.ToInt32(m_DR[6]);
				}
			}
			/// <summary>
			/// Retrieves ISWEBSERVER for the current row.
			/// </summary>
			public int _ISWEBSERVER
			{
				get
				{
					return System.Convert.ToInt32(m_DR[7]);
				}
			}
			/// <summary>
			/// Retrieves ISDATABASESERVER for the current row.
			/// </summary>
			public int _ISDATABASESERVER
			{
				get
				{
					return System.Convert.ToInt32(m_DR[8]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is inserted immediately.
			/// </summary>
			/// <param name="i_ID">the value to be inserted for ID. This value is actually used only if this method call is involved in data publication/replication, otherwise the actual value is generated by the OPERA DB and the supplied value is ignored.</param>
			/// <param name="i_ID_SITE">the value to be inserted for ID_SITE.</param>
			/// <param name="i_NAME">the value to be inserted for NAME.</param>
			/// <param name="i_ADDRESS">the value to be inserted for ADDRESS.</param>
			/// <param name="i_ISSCANNINGSERVER">the value to be inserted for ISSCANNINGSERVER.</param>
			/// <param name="i_ISBATCHSERVER">the value to be inserted for ISBATCHSERVER.</param>
			/// <param name="i_ISDATAPROCESSINGSERVER">the value to be inserted for ISDATAPROCESSINGSERVER.</param>
			/// <param name="i_ISWEBSERVER">the value to be inserted for ISWEBSERVER.</param>
			/// <param name="i_ISDATABASESERVER">the value to be inserted for ISDATABASESERVER.</param>
			/// <returns>the value of ID for the new row.</returns>
			static public long Insert(long i_ID,long i_ID_SITE,string i_NAME,string i_ADDRESS,int i_ISSCANNINGSERVER,int i_ISBATCHSERVER,int i_ISDATAPROCESSINGSERVER,int i_ISWEBSERVER,int i_ISDATABASESERVER)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = i_ID;
				cmd.Parameters[1].Value = i_ID_SITE;
				cmd.Parameters[2].Value = i_NAME;
				cmd.Parameters[3].Value = i_ADDRESS;
				cmd.Parameters[4].Value = i_ISSCANNINGSERVER;
				cmd.Parameters[5].Value = i_ISBATCHSERVER;
				cmd.Parameters[6].Value = i_ISDATAPROCESSINGSERVER;
				cmd.Parameters[7].Value = i_ISWEBSERVER;
				cmd.Parameters[8].Value = i_ISDATABASESERVER;
				cmd.ExecuteNonQuery();
				return SySal.OperaDb.Convert.ToInt64(cmd.Parameters[9].Value);
			}
			/// <summary>
			/// Reads a set of rows from TB_MACHINES and retrieves them into a new TB_MACHINES object.
			/// </summary>
			/// <param name="i_NAME">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="order">the ordering scheme to be applied to returned rows.</param>
			/// <returns>a new instance of the TB_MACHINES class that can be used to read the retrieved data.</returns>
			static public TB_MACHINES SelectPrimaryKey(object i_NAME, OrderBy order)
			{
				string wherestr = "";
				string wtempstr = "";
				if (i_NAME != null)
				{
					if (i_NAME == System.DBNull.Value) wtempstr = "NAME IS NULL";
					else wtempstr = "NAME = " + i_NAME.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (order == OrderBy.Ascending) return SelectWhere(wherestr, "NAME ASC");
				else if (order == OrderBy.Descending) return SelectWhere(wherestr, "NAME DESC");
				return SelectWhere(wherestr, null);
			}
			/// <summary>
			/// Reads a set of rows from TB_MACHINES and retrieves them into a new TB_MACHINES object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the TB_MACHINES class that can be used to read the retrieved data.</returns>
			static public TB_MACHINES SelectWhere(string wherestr, string orderstr)
			{
				TB_MACHINES newobj = new TB_MACHINES();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID,ID_SITE,NAME,ADDRESS,ISSCANNINGSERVER,ISBATCHSERVER,ISDATAPROCESSINGSERVER,ISWEBSERVER,ISDATABASESERVER FROM TB_MACHINES" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO TB_MACHINES (ID,ID_SITE,NAME,ADDRESS,ISSCANNINGSERVER,ISBATCHSERVER,ISDATAPROCESSINGSERVER,ISWEBSERVER,ISDATABASESERVER) VALUES (:p_1,:p_2,:p_3,:p_4,:p_5,:p_6,:p_7,:p_8,:p_9) RETURNING ID INTO :o_9");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Int, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.Int, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_7", SySal.OperaDb.OperaDbType.Int, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_8", SySal.OperaDb.OperaDbType.Int, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_9", SySal.OperaDb.OperaDbType.Int, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("o_9", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Output);
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the TB_MIPBASETRACKS table in the DB.
		/// For data insertion, the Insert and Flush static methods are used. This class inserts data in batches.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class TB_MIPBASETRACKS
		{
			internal TB_MIPBASETRACKS() {}
			System.Data.DataRowCollection m_DRC;
			const int ArraySize = 100;
			static int index = 0;
			/// <summary>
			/// Since this class uses batch insertion, this method must be called to ensure that all rows are actually inserted into TB_MIPBASETRACKS. Failure to do so will result in incomplete writes.
			/// </summary>
			static public void Flush() { if (cmd == null || index == 0) return; cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			private static long [] a_ID_EVENTBRICK = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_EVENTBRICK for the current row.
			/// </summary>
			public long _ID_EVENTBRICK
			{
				get
				{
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			private static long [] a_ID_ZONE = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_ZONE for the current row.
			/// </summary>
			public long _ID_ZONE
			{
				get
				{
					return System.Convert.ToInt64(m_DR[1]);
				}
			}
			private static long [] a_ID = new long[ArraySize];
			/// <summary>
			/// Retrieves ID for the current row.
			/// </summary>
			public long _ID
			{
				get
				{
					return System.Convert.ToInt64(m_DR[2]);
				}
			}
			private static double [] a_POSX = new double[ArraySize];
			/// <summary>
			/// Retrieves POSX for the current row.
			/// </summary>
			public double _POSX
			{
				get
				{
					return System.Convert.ToDouble(m_DR[3]);
				}
			}
			private static double [] a_POSY = new double[ArraySize];
			/// <summary>
			/// Retrieves POSY for the current row.
			/// </summary>
			public double _POSY
			{
				get
				{
					return System.Convert.ToDouble(m_DR[4]);
				}
			}
			private static double [] a_SLOPEX = new double[ArraySize];
			/// <summary>
			/// Retrieves SLOPEX for the current row.
			/// </summary>
			public double _SLOPEX
			{
				get
				{
					return System.Convert.ToDouble(m_DR[5]);
				}
			}
			private static double [] a_SLOPEY = new double[ArraySize];
			/// <summary>
			/// Retrieves SLOPEY for the current row.
			/// </summary>
			public double _SLOPEY
			{
				get
				{
					return System.Convert.ToDouble(m_DR[6]);
				}
			}
			private static object [] a_GRAINS = new object[ArraySize];
			/// <summary>
			/// Retrieves GRAINS for the current row. The return value can be System.DBNull.Value or a value that can be cast to int.
			/// </summary>
			public object _GRAINS
			{
				get
				{
					if (m_DR[7] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt32(m_DR[7]);
				}
			}
			private static object [] a_AREASUM = new object[ArraySize];
			/// <summary>
			/// Retrieves AREASUM for the current row. The return value can be System.DBNull.Value or a value that can be cast to int.
			/// </summary>
			public object _AREASUM
			{
				get
				{
					if (m_DR[8] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt32(m_DR[8]);
				}
			}
			private static object [] a_PH = new object[ArraySize];
			/// <summary>
			/// Retrieves PH for the current row. The return value can be System.DBNull.Value or a value that can be cast to double.
			/// </summary>
			public object _PH
			{
				get
				{
					if (m_DR[9] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDouble(m_DR[9]);
				}
			}
			private static double [] a_SIGMA = new double[ArraySize];
			/// <summary>
			/// Retrieves SIGMA for the current row.
			/// </summary>
			public double _SIGMA
			{
				get
				{
					return System.Convert.ToDouble(m_DR[10]);
				}
			}
			private static int [] a_ID_DOWNSIDE = new int[ArraySize];
			/// <summary>
			/// Retrieves ID_DOWNSIDE for the current row.
			/// </summary>
			public int _ID_DOWNSIDE
			{
				get
				{
					return System.Convert.ToInt32(m_DR[11]);
				}
			}
			private static long [] a_ID_DOWNTRACK = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_DOWNTRACK for the current row.
			/// </summary>
			public long _ID_DOWNTRACK
			{
				get
				{
					return System.Convert.ToInt64(m_DR[12]);
				}
			}
			private static int [] a_ID_UPSIDE = new int[ArraySize];
			/// <summary>
			/// Retrieves ID_UPSIDE for the current row.
			/// </summary>
			public int _ID_UPSIDE
			{
				get
				{
					return System.Convert.ToInt32(m_DR[13]);
				}
			}
			private static long [] a_ID_UPTRACK = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_UPTRACK for the current row.
			/// </summary>
			public long _ID_UPTRACK
			{
				get
				{
					return System.Convert.ToInt64(m_DR[14]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is actually inserted later, in a batch insertion command. The Flush method must be called to ensure all rows are actually written.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">the value to be inserted for ID_EVENTBRICK.</param>
			/// <param name="i_ID_ZONE">the value to be inserted for ID_ZONE.</param>
			/// <param name="i_ID">the value to be inserted for ID.</param>
			/// <param name="i_POSX">the value to be inserted for POSX.</param>
			/// <param name="i_POSY">the value to be inserted for POSY.</param>
			/// <param name="i_SLOPEX">the value to be inserted for SLOPEX.</param>
			/// <param name="i_SLOPEY">the value to be inserted for SLOPEY.</param>
			/// <param name="i_GRAINS">the value to be inserted for GRAINS. The value for this parameter can be int or System.DBNull.Value.</param>
			/// <param name="i_AREASUM">the value to be inserted for AREASUM. The value for this parameter can be int or System.DBNull.Value.</param>
			/// <param name="i_PH">the value to be inserted for PH. The value for this parameter can be double or System.DBNull.Value.</param>
			/// <param name="i_SIGMA">the value to be inserted for SIGMA.</param>
			/// <param name="i_ID_DOWNSIDE">the value to be inserted for ID_DOWNSIDE.</param>
			/// <param name="i_ID_DOWNTRACK">the value to be inserted for ID_DOWNTRACK.</param>
			/// <param name="i_ID_UPSIDE">the value to be inserted for ID_UPSIDE.</param>
			/// <param name="i_ID_UPTRACK">the value to be inserted for ID_UPTRACK.</param>
			static public void Insert(long i_ID_EVENTBRICK,long i_ID_ZONE,long i_ID,double i_POSX,double i_POSY,double i_SLOPEX,double i_SLOPEY,object i_GRAINS,object i_AREASUM,object i_PH,double i_SIGMA,int i_ID_DOWNSIDE,long i_ID_DOWNTRACK,int i_ID_UPSIDE,long i_ID_UPTRACK)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				a_ID_EVENTBRICK[index] = i_ID_EVENTBRICK;
				a_ID_ZONE[index] = i_ID_ZONE;
				a_ID[index] = i_ID;
				a_POSX[index] = i_POSX;
				a_POSY[index] = i_POSY;
				a_SLOPEX[index] = i_SLOPEX;
				a_SLOPEY[index] = i_SLOPEY;
				a_GRAINS[index] = i_GRAINS;
				a_AREASUM[index] = i_AREASUM;
				a_PH[index] = i_PH;
				a_SIGMA[index] = i_SIGMA;
				a_ID_DOWNSIDE[index] = i_ID_DOWNSIDE;
				a_ID_DOWNTRACK[index] = i_ID_DOWNTRACK;
				a_ID_UPSIDE[index] = i_ID_UPSIDE;
				a_ID_UPTRACK[index] = i_ID_UPTRACK;
				if (++index >= ArraySize) { cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			}
			/// <summary>
			/// Reads a set of rows from TB_MIPBASETRACKS and retrieves them into a new TB_MIPBASETRACKS object.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_ZONE">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="order">the ordering scheme to be applied to returned rows. This applies to all columns in the primary key.</param>
			/// <returns>a new instance of the TB_MIPBASETRACKS class that can be used to read the retrieved data.</returns>
			static public TB_MIPBASETRACKS SelectPrimaryKey(object i_ID_EVENTBRICK,object i_ID_ZONE,object i_ID, OrderBy order)
			{
				string wherestr = "";
				string wtempstr = "";
				if (i_ID_EVENTBRICK != null)
				{
					if (i_ID_EVENTBRICK == System.DBNull.Value) wtempstr = "ID_EVENTBRICK IS NULL";
					else wtempstr = "ID_EVENTBRICK = " + i_ID_EVENTBRICK.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_ZONE != null)
				{
					if (i_ID_ZONE == System.DBNull.Value) wtempstr = "ID_ZONE IS NULL";
					else wtempstr = "ID_ZONE = " + i_ID_ZONE.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID != null)
				{
					if (i_ID == System.DBNull.Value) wtempstr = "ID IS NULL";
					else wtempstr = "ID = " + i_ID.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (order == OrderBy.Ascending) return SelectWhere(wherestr, "ID_EVENTBRICK ASC,ID_ZONE ASC,ID ASC");
				else if (order == OrderBy.Descending) return SelectWhere(wherestr, "ID_EVENTBRICK DESC,ID_ZONE DESC,ID DESC");
				return SelectWhere(wherestr, null);
			}
			/// <summary>
			/// Reads a set of rows from TB_MIPBASETRACKS and retrieves them into a new TB_MIPBASETRACKS object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the TB_MIPBASETRACKS class that can be used to read the retrieved data.</returns>
			static public TB_MIPBASETRACKS SelectWhere(string wherestr, string orderstr)
			{
				TB_MIPBASETRACKS newobj = new TB_MIPBASETRACKS();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_EVENTBRICK,ID_ZONE,ID,POSX,POSY,SLOPEX,SLOPEY,GRAINS,AREASUM,PH,SIGMA,ID_DOWNSIDE,ID_DOWNTRACK,ID_UPSIDE,ID_UPTRACK FROM TB_MIPBASETRACKS" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO TB_MIPBASETRACKS (ID_EVENTBRICK,ID_ZONE,ID,POSX,POSY,SLOPEX,SLOPEY,GRAINS,AREASUM,PH,SIGMA,ID_DOWNSIDE,ID_DOWNTRACK,ID_UPSIDE,ID_UPTRACK) VALUES (:p_1,:p_2,:p_3,:p_4,:p_5,:p_6,:p_7,:p_8,:p_9,:p_10,:p_11,:p_12,:p_13,:p_14,:p_15)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_EVENTBRICK;
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_ZONE;
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID;
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_POSX;
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_POSY;
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_SLOPEX;
				newcmd.Parameters.Add("p_7", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_SLOPEY;
				newcmd.Parameters.Add("p_8", SySal.OperaDb.OperaDbType.Int, System.Data.ParameterDirection.Input).Value = a_GRAINS;
				newcmd.Parameters.Add("p_9", SySal.OperaDb.OperaDbType.Int, System.Data.ParameterDirection.Input).Value = a_AREASUM;
				newcmd.Parameters.Add("p_10", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_PH;
				newcmd.Parameters.Add("p_11", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_SIGMA;
				newcmd.Parameters.Add("p_12", SySal.OperaDb.OperaDbType.Int, System.Data.ParameterDirection.Input).Value = a_ID_DOWNSIDE;
				newcmd.Parameters.Add("p_13", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_DOWNTRACK;
				newcmd.Parameters.Add("p_14", SySal.OperaDb.OperaDbType.Int, System.Data.ParameterDirection.Input).Value = a_ID_UPSIDE;
				newcmd.Parameters.Add("p_15", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_UPTRACK;
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the TB_MIPMICROTRACKS table in the DB.
		/// For data insertion, the Insert and Flush static methods are used. This class inserts data in batches.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class TB_MIPMICROTRACKS
		{
			internal TB_MIPMICROTRACKS() {}
			System.Data.DataRowCollection m_DRC;
			const int ArraySize = 100;
			static int index = 0;
			/// <summary>
			/// Since this class uses batch insertion, this method must be called to ensure that all rows are actually inserted into TB_MIPMICROTRACKS. Failure to do so will result in incomplete writes.
			/// </summary>
			static public void Flush() { if (cmd == null || index == 0) return; cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			private static long [] a_ID_EVENTBRICK = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_EVENTBRICK for the current row.
			/// </summary>
			public long _ID_EVENTBRICK
			{
				get
				{
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			private static long [] a_ID_ZONE = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_ZONE for the current row.
			/// </summary>
			public long _ID_ZONE
			{
				get
				{
					return System.Convert.ToInt64(m_DR[1]);
				}
			}
			private static int [] a_SIDE = new int[ArraySize];
			/// <summary>
			/// Retrieves SIDE for the current row.
			/// </summary>
			public int _SIDE
			{
				get
				{
					return System.Convert.ToInt32(m_DR[2]);
				}
			}
			private static long [] a_ID = new long[ArraySize];
			/// <summary>
			/// Retrieves ID for the current row.
			/// </summary>
			public long _ID
			{
				get
				{
					return System.Convert.ToInt64(m_DR[3]);
				}
			}
			private static double [] a_POSX = new double[ArraySize];
			/// <summary>
			/// Retrieves POSX for the current row.
			/// </summary>
			public double _POSX
			{
				get
				{
					return System.Convert.ToDouble(m_DR[4]);
				}
			}
			private static double [] a_POSY = new double[ArraySize];
			/// <summary>
			/// Retrieves POSY for the current row.
			/// </summary>
			public double _POSY
			{
				get
				{
					return System.Convert.ToDouble(m_DR[5]);
				}
			}
			private static double [] a_SLOPEX = new double[ArraySize];
			/// <summary>
			/// Retrieves SLOPEX for the current row.
			/// </summary>
			public double _SLOPEX
			{
				get
				{
					return System.Convert.ToDouble(m_DR[6]);
				}
			}
			private static double [] a_SLOPEY = new double[ArraySize];
			/// <summary>
			/// Retrieves SLOPEY for the current row.
			/// </summary>
			public double _SLOPEY
			{
				get
				{
					return System.Convert.ToDouble(m_DR[7]);
				}
			}
			private static object [] a_GRAINS = new object[ArraySize];
			/// <summary>
			/// Retrieves GRAINS for the current row. The return value can be System.DBNull.Value or a value that can be cast to int.
			/// </summary>
			public object _GRAINS
			{
				get
				{
					if (m_DR[8] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt32(m_DR[8]);
				}
			}
			private static object [] a_AREASUM = new object[ArraySize];
			/// <summary>
			/// Retrieves AREASUM for the current row. The return value can be System.DBNull.Value or a value that can be cast to int.
			/// </summary>
			public object _AREASUM
			{
				get
				{
					if (m_DR[9] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt32(m_DR[9]);
				}
			}
			private static object [] a_PH = new object[ArraySize];
			/// <summary>
			/// Retrieves PH for the current row. The return value can be System.DBNull.Value or a value that can be cast to double.
			/// </summary>
			public object _PH
			{
				get
				{
					if (m_DR[10] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDouble(m_DR[10]);
				}
			}
			private static double [] a_SIGMA = new double[ArraySize];
			/// <summary>
			/// Retrieves SIGMA for the current row.
			/// </summary>
			public double _SIGMA
			{
				get
				{
					return System.Convert.ToDouble(m_DR[11]);
				}
			}
			private static long [] a_ID_VIEW = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_VIEW for the current row.
			/// </summary>
			public long _ID_VIEW
			{
				get
				{
					return System.Convert.ToInt64(m_DR[12]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is actually inserted later, in a batch insertion command. The Flush method must be called to ensure all rows are actually written.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">the value to be inserted for ID_EVENTBRICK.</param>
			/// <param name="i_ID_ZONE">the value to be inserted for ID_ZONE.</param>
			/// <param name="i_SIDE">the value to be inserted for SIDE.</param>
			/// <param name="i_ID">the value to be inserted for ID.</param>
			/// <param name="i_POSX">the value to be inserted for POSX.</param>
			/// <param name="i_POSY">the value to be inserted for POSY.</param>
			/// <param name="i_SLOPEX">the value to be inserted for SLOPEX.</param>
			/// <param name="i_SLOPEY">the value to be inserted for SLOPEY.</param>
			/// <param name="i_GRAINS">the value to be inserted for GRAINS. The value for this parameter can be int or System.DBNull.Value.</param>
			/// <param name="i_AREASUM">the value to be inserted for AREASUM. The value for this parameter can be int or System.DBNull.Value.</param>
			/// <param name="i_PH">the value to be inserted for PH. The value for this parameter can be double or System.DBNull.Value.</param>
			/// <param name="i_SIGMA">the value to be inserted for SIGMA.</param>
			/// <param name="i_ID_VIEW">the value to be inserted for ID_VIEW.</param>
			static public void Insert(long i_ID_EVENTBRICK,long i_ID_ZONE,int i_SIDE,long i_ID,double i_POSX,double i_POSY,double i_SLOPEX,double i_SLOPEY,object i_GRAINS,object i_AREASUM,object i_PH,double i_SIGMA,long i_ID_VIEW)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				a_ID_EVENTBRICK[index] = i_ID_EVENTBRICK;
				a_ID_ZONE[index] = i_ID_ZONE;
				a_SIDE[index] = i_SIDE;
				a_ID[index] = i_ID;
				a_POSX[index] = i_POSX;
				a_POSY[index] = i_POSY;
				a_SLOPEX[index] = i_SLOPEX;
				a_SLOPEY[index] = i_SLOPEY;
				a_GRAINS[index] = i_GRAINS;
				a_AREASUM[index] = i_AREASUM;
				a_PH[index] = i_PH;
				a_SIGMA[index] = i_SIGMA;
				a_ID_VIEW[index] = i_ID_VIEW;
				if (++index >= ArraySize) { cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			}
			/// <summary>
			/// Reads a set of rows from TB_MIPMICROTRACKS and retrieves them into a new TB_MIPMICROTRACKS object.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_ZONE">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_SIDE">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="order">the ordering scheme to be applied to returned rows. This applies to all columns in the primary key.</param>
			/// <returns>a new instance of the TB_MIPMICROTRACKS class that can be used to read the retrieved data.</returns>
			static public TB_MIPMICROTRACKS SelectPrimaryKey(object i_ID_EVENTBRICK,object i_ID_ZONE,object i_SIDE,object i_ID, OrderBy order)
			{
				string wherestr = "";
				string wtempstr = "";
				if (i_ID_EVENTBRICK != null)
				{
					if (i_ID_EVENTBRICK == System.DBNull.Value) wtempstr = "ID_EVENTBRICK IS NULL";
					else wtempstr = "ID_EVENTBRICK = " + i_ID_EVENTBRICK.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_ZONE != null)
				{
					if (i_ID_ZONE == System.DBNull.Value) wtempstr = "ID_ZONE IS NULL";
					else wtempstr = "ID_ZONE = " + i_ID_ZONE.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_SIDE != null)
				{
					if (i_SIDE == System.DBNull.Value) wtempstr = "SIDE IS NULL";
					else wtempstr = "SIDE = " + i_SIDE.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID != null)
				{
					if (i_ID == System.DBNull.Value) wtempstr = "ID IS NULL";
					else wtempstr = "ID = " + i_ID.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (order == OrderBy.Ascending) return SelectWhere(wherestr, "ID_EVENTBRICK ASC,ID_ZONE ASC,SIDE ASC,ID ASC");
				else if (order == OrderBy.Descending) return SelectWhere(wherestr, "ID_EVENTBRICK DESC,ID_ZONE DESC,SIDE DESC,ID DESC");
				return SelectWhere(wherestr, null);
			}
			/// <summary>
			/// Reads a set of rows from TB_MIPMICROTRACKS and retrieves them into a new TB_MIPMICROTRACKS object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the TB_MIPMICROTRACKS class that can be used to read the retrieved data.</returns>
			static public TB_MIPMICROTRACKS SelectWhere(string wherestr, string orderstr)
			{
				TB_MIPMICROTRACKS newobj = new TB_MIPMICROTRACKS();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_EVENTBRICK,ID_ZONE,SIDE,ID,POSX,POSY,SLOPEX,SLOPEY,GRAINS,AREASUM,PH,SIGMA,ID_VIEW FROM TB_MIPMICROTRACKS" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO TB_MIPMICROTRACKS (ID_EVENTBRICK,ID_ZONE,SIDE,ID,POSX,POSY,SLOPEX,SLOPEY,GRAINS,AREASUM,PH,SIGMA,ID_VIEW) VALUES (:p_1,:p_2,:p_3,:p_4,:p_5,:p_6,:p_7,:p_8,:p_9,:p_10,:p_11,:p_12,:p_13)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_EVENTBRICK;
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_ZONE;
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Int, System.Data.ParameterDirection.Input).Value = a_SIDE;
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID;
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_POSX;
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_POSY;
				newcmd.Parameters.Add("p_7", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_SLOPEX;
				newcmd.Parameters.Add("p_8", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_SLOPEY;
				newcmd.Parameters.Add("p_9", SySal.OperaDb.OperaDbType.Int, System.Data.ParameterDirection.Input).Value = a_GRAINS;
				newcmd.Parameters.Add("p_10", SySal.OperaDb.OperaDbType.Int, System.Data.ParameterDirection.Input).Value = a_AREASUM;
				newcmd.Parameters.Add("p_11", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_PH;
				newcmd.Parameters.Add("p_12", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_SIGMA;
				newcmd.Parameters.Add("p_13", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_VIEW;
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the TB_PACKAGED_SW table in the DB.
		/// For data insertion, the Insert and Flush static methods are used. This class inserts data in batches.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class TB_PACKAGED_SW
		{
			internal TB_PACKAGED_SW() {}
			System.Data.DataRowCollection m_DRC;
			const int ArraySize = 100;
			static int index = 0;
			/// <summary>
			/// Since this class uses batch insertion, this method must be called to ensure that all rows are actually inserted into TB_PACKAGED_SW. Failure to do so will result in incomplete writes.
			/// </summary>
			static public void Flush() { if (cmd == null || index == 0) return; cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			private static string [] a_PACKAGE_NAME = new string[ArraySize];
			/// <summary>
			/// Retrieves PACKAGE_NAME for the current row.
			/// </summary>
			public string _PACKAGE_NAME
			{
				get
				{
					return System.Convert.ToString(m_DR[0]);
				}
			}
			private static long [] a_MAJOR_VERSION = new long[ArraySize];
			/// <summary>
			/// Retrieves MAJOR_VERSION for the current row.
			/// </summary>
			public long _MAJOR_VERSION
			{
				get
				{
					return System.Convert.ToInt64(m_DR[1]);
				}
			}
			private static long [] a_MINOR_VERSION = new long[ArraySize];
			/// <summary>
			/// Retrieves MINOR_VERSION for the current row.
			/// </summary>
			public long _MINOR_VERSION
			{
				get
				{
					return System.Convert.ToInt64(m_DR[2]);
				}
			}
			private static long [] a_EDITION = new long[ArraySize];
			/// <summary>
			/// Retrieves EDITION for the current row.
			/// </summary>
			public long _EDITION
			{
				get
				{
					return System.Convert.ToInt64(m_DR[3]);
				}
			}
			private static string [] a_SUBDIRECTORY = new string[ArraySize];
			/// <summary>
			/// Retrieves SUBDIRECTORY for the current row.
			/// </summary>
			public string _SUBDIRECTORY
			{
				get
				{
					return System.Convert.ToString(m_DR[4]);
				}
			}
			private static long [] a_COMMAND_SHARP_ = new long[ArraySize];
			/// <summary>
			/// Retrieves COMMAND# for the current row.
			/// </summary>
			public long _COMMAND_SHARP_
			{
				get
				{
					return System.Convert.ToInt64(m_DR[5]);
				}
			}
			private static object [] a_COMMAND_TEXT = new object[ArraySize];
			/// <summary>
			/// Retrieves COMMAND_TEXT for the current row. The return value can be System.DBNull.Value or a value that can be cast to string.
			/// </summary>
			public object _COMMAND_TEXT
			{
				get
				{
					if (m_DR[6] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToString(m_DR[6]);
				}
			}
			private static string [] a_COMPONENT_NAME = new string[ArraySize];
			/// <summary>
			/// Retrieves COMPONENT_NAME for the current row.
			/// </summary>
			public string _COMPONENT_NAME
			{
				get
				{
					return System.Convert.ToString(m_DR[7]);
				}
			}
			private static object [] a_IMAGE = new object[ArraySize];
			/// <summary>
			/// Retrieves IMAGE for the current row. The return value can be System.DBNull.Value or a value that can be cast to byte [].
			/// </summary>
			public object _IMAGE
			{
				get
				{
					if (m_DR[8] == System.DBNull.Value) return System.DBNull.Value;
					return SySal.OperaDb.Convert.ToBytes(m_DR[8]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is actually inserted later, in a batch insertion command. The Flush method must be called to ensure all rows are actually written.
			/// </summary>
			/// <param name="i_PACKAGE_NAME">the value to be inserted for PACKAGE_NAME.</param>
			/// <param name="i_MAJOR_VERSION">the value to be inserted for MAJOR_VERSION.</param>
			/// <param name="i_MINOR_VERSION">the value to be inserted for MINOR_VERSION.</param>
			/// <param name="i_EDITION">the value to be inserted for EDITION.</param>
			/// <param name="i_SUBDIRECTORY">the value to be inserted for SUBDIRECTORY.</param>
			/// <param name="i_COMMAND_SHARP_">the value to be inserted for COMMAND#.</param>
			/// <param name="i_COMMAND_TEXT">the value to be inserted for COMMAND_TEXT. The value for this parameter can be string or System.DBNull.Value.</param>
			/// <param name="i_COMPONENT_NAME">the value to be inserted for COMPONENT_NAME.</param>
			/// <param name="i_IMAGE">the value to be inserted for IMAGE. The value for this parameter can be byte [] or System.DBNull.Value.</param>
			static public void Insert(string i_PACKAGE_NAME,long i_MAJOR_VERSION,long i_MINOR_VERSION,long i_EDITION,string i_SUBDIRECTORY,long i_COMMAND_SHARP_,object i_COMMAND_TEXT,string i_COMPONENT_NAME,object i_IMAGE)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				a_PACKAGE_NAME[index] = i_PACKAGE_NAME;
				a_MAJOR_VERSION[index] = i_MAJOR_VERSION;
				a_MINOR_VERSION[index] = i_MINOR_VERSION;
				a_EDITION[index] = i_EDITION;
				a_SUBDIRECTORY[index] = i_SUBDIRECTORY;
				a_COMMAND_SHARP_[index] = i_COMMAND_SHARP_;
				a_COMMAND_TEXT[index] = i_COMMAND_TEXT;
				a_COMPONENT_NAME[index] = i_COMPONENT_NAME;
				a_IMAGE[index] = i_IMAGE;
				if (++index >= ArraySize) { cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			}
			/// <summary>
			/// Reads a set of rows from TB_PACKAGED_SW and retrieves them into a new TB_PACKAGED_SW object.
			/// </summary>
			/// <param name="i_PACKAGE_NAME">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_MAJOR_VERSION">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_MINOR_VERSION">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_EDITION">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_SUBDIRECTORY">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_COMMAND_SHARP_">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_COMPONENT_NAME">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="order">the ordering scheme to be applied to returned rows. This applies to all columns in the primary key.</param>
			/// <returns>a new instance of the TB_PACKAGED_SW class that can be used to read the retrieved data.</returns>
			static public TB_PACKAGED_SW SelectPrimaryKey(object i_PACKAGE_NAME,object i_MAJOR_VERSION,object i_MINOR_VERSION,object i_EDITION,object i_SUBDIRECTORY,object i_COMMAND_SHARP_,object i_COMPONENT_NAME, OrderBy order)
			{
				string wherestr = "";
				string wtempstr = "";
				if (i_PACKAGE_NAME != null)
				{
					if (i_PACKAGE_NAME == System.DBNull.Value) wtempstr = "PACKAGE_NAME IS NULL";
					else wtempstr = "PACKAGE_NAME = " + i_PACKAGE_NAME.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_MAJOR_VERSION != null)
				{
					if (i_MAJOR_VERSION == System.DBNull.Value) wtempstr = "MAJOR_VERSION IS NULL";
					else wtempstr = "MAJOR_VERSION = " + i_MAJOR_VERSION.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_MINOR_VERSION != null)
				{
					if (i_MINOR_VERSION == System.DBNull.Value) wtempstr = "MINOR_VERSION IS NULL";
					else wtempstr = "MINOR_VERSION = " + i_MINOR_VERSION.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_EDITION != null)
				{
					if (i_EDITION == System.DBNull.Value) wtempstr = "EDITION IS NULL";
					else wtempstr = "EDITION = " + i_EDITION.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_SUBDIRECTORY != null)
				{
					if (i_SUBDIRECTORY == System.DBNull.Value) wtempstr = "SUBDIRECTORY IS NULL";
					else wtempstr = "SUBDIRECTORY = " + i_SUBDIRECTORY.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_COMMAND_SHARP_ != null)
				{
					if (i_COMMAND_SHARP_ == System.DBNull.Value) wtempstr = "COMMAND# IS NULL";
					else wtempstr = "COMMAND# = " + i_COMMAND_SHARP_.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_COMPONENT_NAME != null)
				{
					if (i_COMPONENT_NAME == System.DBNull.Value) wtempstr = "COMPONENT_NAME IS NULL";
					else wtempstr = "COMPONENT_NAME = " + i_COMPONENT_NAME.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (order == OrderBy.Ascending) return SelectWhere(wherestr, "PACKAGE_NAME ASC,MAJOR_VERSION ASC,MINOR_VERSION ASC,EDITION ASC,SUBDIRECTORY ASC,COMMAND# ASC,COMPONENT_NAME ASC");
				else if (order == OrderBy.Descending) return SelectWhere(wherestr, "PACKAGE_NAME DESC,MAJOR_VERSION DESC,MINOR_VERSION DESC,EDITION DESC,SUBDIRECTORY DESC,COMMAND# DESC,COMPONENT_NAME DESC");
				return SelectWhere(wherestr, null);
			}
			/// <summary>
			/// Reads a set of rows from TB_PACKAGED_SW and retrieves them into a new TB_PACKAGED_SW object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the TB_PACKAGED_SW class that can be used to read the retrieved data.</returns>
			static public TB_PACKAGED_SW SelectWhere(string wherestr, string orderstr)
			{
				TB_PACKAGED_SW newobj = new TB_PACKAGED_SW();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT PACKAGE_NAME,MAJOR_VERSION,MINOR_VERSION,EDITION,SUBDIRECTORY,COMMAND#,COMMAND_TEXT,COMPONENT_NAME,IMAGE FROM TB_PACKAGED_SW" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO TB_PACKAGED_SW (PACKAGE_NAME,MAJOR_VERSION,MINOR_VERSION,EDITION,SUBDIRECTORY,COMMAND#,COMMAND_TEXT,COMPONENT_NAME,IMAGE) VALUES (:p_1,:p_2,:p_3,:p_4,:p_5,:p_6,:p_7,:p_8,:p_9)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input).Value = a_PACKAGE_NAME;
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_MAJOR_VERSION;
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_MINOR_VERSION;
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_EDITION;
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input).Value = a_SUBDIRECTORY;
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_COMMAND_SHARP_;
				newcmd.Parameters.Add("p_7", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input).Value = a_COMMAND_TEXT;
				newcmd.Parameters.Add("p_8", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input).Value = a_COMPONENT_NAME;
				newcmd.Parameters.Add("p_9", SySal.OperaDb.OperaDbType.BLOB, System.Data.ParameterDirection.Input).Value = a_IMAGE;
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the TB_PATTERN_MATCH table in the DB.
		/// For data insertion, the Insert and Flush static methods are used. This class inserts data in batches.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class TB_PATTERN_MATCH
		{
			internal TB_PATTERN_MATCH() {}
			System.Data.DataRowCollection m_DRC;
			const int ArraySize = 100;
			static int index = 0;
			/// <summary>
			/// Since this class uses batch insertion, this method must be called to ensure that all rows are actually inserted into TB_PATTERN_MATCH. Failure to do so will result in incomplete writes.
			/// </summary>
			static public void Flush() { if (cmd == null || index == 0) return; cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			private static long [] a_ID_EVENTBRICK = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_EVENTBRICK for the current row.
			/// </summary>
			public long _ID_EVENTBRICK
			{
				get
				{
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			private static long [] a_ID_FIRSTZONE = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_FIRSTZONE for the current row.
			/// </summary>
			public long _ID_FIRSTZONE
			{
				get
				{
					return System.Convert.ToInt64(m_DR[1]);
				}
			}
			private static long [] a_ID_SECONDZONE = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_SECONDZONE for the current row.
			/// </summary>
			public long _ID_SECONDZONE
			{
				get
				{
					return System.Convert.ToInt64(m_DR[2]);
				}
			}
			private static long [] a_ID_INFIRSTZONE = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_INFIRSTZONE for the current row.
			/// </summary>
			public long _ID_INFIRSTZONE
			{
				get
				{
					return System.Convert.ToInt64(m_DR[3]);
				}
			}
			private static long [] a_ID_INSECONDZONE = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_INSECONDZONE for the current row.
			/// </summary>
			public long _ID_INSECONDZONE
			{
				get
				{
					return System.Convert.ToInt64(m_DR[4]);
				}
			}
			private static long [] a_ID_PROCESSOPERATION = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_PROCESSOPERATION for the current row.
			/// </summary>
			public long _ID_PROCESSOPERATION
			{
				get
				{
					return System.Convert.ToInt64(m_DR[5]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is actually inserted later, in a batch insertion command. The Flush method must be called to ensure all rows are actually written.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">the value to be inserted for ID_EVENTBRICK.</param>
			/// <param name="i_ID_FIRSTZONE">the value to be inserted for ID_FIRSTZONE.</param>
			/// <param name="i_ID_SECONDZONE">the value to be inserted for ID_SECONDZONE.</param>
			/// <param name="i_ID_INFIRSTZONE">the value to be inserted for ID_INFIRSTZONE.</param>
			/// <param name="i_ID_INSECONDZONE">the value to be inserted for ID_INSECONDZONE.</param>
			/// <param name="i_ID_PROCESSOPERATION">the value to be inserted for ID_PROCESSOPERATION.</param>
			static public void Insert(long i_ID_EVENTBRICK,long i_ID_FIRSTZONE,long i_ID_SECONDZONE,long i_ID_INFIRSTZONE,long i_ID_INSECONDZONE,long i_ID_PROCESSOPERATION)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				a_ID_EVENTBRICK[index] = i_ID_EVENTBRICK;
				a_ID_FIRSTZONE[index] = i_ID_FIRSTZONE;
				a_ID_SECONDZONE[index] = i_ID_SECONDZONE;
				a_ID_INFIRSTZONE[index] = i_ID_INFIRSTZONE;
				a_ID_INSECONDZONE[index] = i_ID_INSECONDZONE;
				a_ID_PROCESSOPERATION[index] = i_ID_PROCESSOPERATION;
				if (++index >= ArraySize) { cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			}
			/// <summary>
			/// Reads a set of rows from TB_PATTERN_MATCH and retrieves them into a new TB_PATTERN_MATCH object.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_FIRSTZONE">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_SECONDZONE">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_INFIRSTZONE">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_INSECONDZONE">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_PROCESSOPERATION">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="order">the ordering scheme to be applied to returned rows. This applies to all columns in the primary key.</param>
			/// <returns>a new instance of the TB_PATTERN_MATCH class that can be used to read the retrieved data.</returns>
			static public TB_PATTERN_MATCH SelectPrimaryKey(object i_ID_EVENTBRICK,object i_ID_FIRSTZONE,object i_ID_SECONDZONE,object i_ID_INFIRSTZONE,object i_ID_INSECONDZONE,object i_ID_PROCESSOPERATION, OrderBy order)
			{
				string wherestr = "";
				string wtempstr = "";
				if (i_ID_EVENTBRICK != null)
				{
					if (i_ID_EVENTBRICK == System.DBNull.Value) wtempstr = "ID_EVENTBRICK IS NULL";
					else wtempstr = "ID_EVENTBRICK = " + i_ID_EVENTBRICK.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_FIRSTZONE != null)
				{
					if (i_ID_FIRSTZONE == System.DBNull.Value) wtempstr = "ID_FIRSTZONE IS NULL";
					else wtempstr = "ID_FIRSTZONE = " + i_ID_FIRSTZONE.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_SECONDZONE != null)
				{
					if (i_ID_SECONDZONE == System.DBNull.Value) wtempstr = "ID_SECONDZONE IS NULL";
					else wtempstr = "ID_SECONDZONE = " + i_ID_SECONDZONE.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_INFIRSTZONE != null)
				{
					if (i_ID_INFIRSTZONE == System.DBNull.Value) wtempstr = "ID_INFIRSTZONE IS NULL";
					else wtempstr = "ID_INFIRSTZONE = " + i_ID_INFIRSTZONE.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_INSECONDZONE != null)
				{
					if (i_ID_INSECONDZONE == System.DBNull.Value) wtempstr = "ID_INSECONDZONE IS NULL";
					else wtempstr = "ID_INSECONDZONE = " + i_ID_INSECONDZONE.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_PROCESSOPERATION != null)
				{
					if (i_ID_PROCESSOPERATION == System.DBNull.Value) wtempstr = "ID_PROCESSOPERATION IS NULL";
					else wtempstr = "ID_PROCESSOPERATION = " + i_ID_PROCESSOPERATION.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (order == OrderBy.Ascending) return SelectWhere(wherestr, "ID_EVENTBRICK ASC,ID_FIRSTZONE ASC,ID_SECONDZONE ASC,ID_INFIRSTZONE ASC,ID_INSECONDZONE ASC,ID_PROCESSOPERATION ASC");
				else if (order == OrderBy.Descending) return SelectWhere(wherestr, "ID_EVENTBRICK DESC,ID_FIRSTZONE DESC,ID_SECONDZONE DESC,ID_INFIRSTZONE DESC,ID_INSECONDZONE DESC,ID_PROCESSOPERATION DESC");
				return SelectWhere(wherestr, null);
			}
			/// <summary>
			/// Reads a set of rows from TB_PATTERN_MATCH and retrieves them into a new TB_PATTERN_MATCH object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the TB_PATTERN_MATCH class that can be used to read the retrieved data.</returns>
			static public TB_PATTERN_MATCH SelectWhere(string wherestr, string orderstr)
			{
				TB_PATTERN_MATCH newobj = new TB_PATTERN_MATCH();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_EVENTBRICK,ID_FIRSTZONE,ID_SECONDZONE,ID_INFIRSTZONE,ID_INSECONDZONE,ID_PROCESSOPERATION FROM TB_PATTERN_MATCH" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO TB_PATTERN_MATCH (ID_EVENTBRICK,ID_FIRSTZONE,ID_SECONDZONE,ID_INFIRSTZONE,ID_INSECONDZONE,ID_PROCESSOPERATION) VALUES (:p_1,:p_2,:p_3,:p_4,:p_5,:p_6)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_EVENTBRICK;
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_FIRSTZONE;
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_SECONDZONE;
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_INFIRSTZONE;
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_INSECONDZONE;
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_PROCESSOPERATION;
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the TB_PEANUT_BRICKALIGN table in the DB.
		/// For data insertion, the Insert and Flush static methods are used. This class inserts data in batches.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class TB_PEANUT_BRICKALIGN
		{
			internal TB_PEANUT_BRICKALIGN() {}
			System.Data.DataRowCollection m_DRC;
			const int ArraySize = 100;
			static int index = 0;
			/// <summary>
			/// Since this class uses batch insertion, this method must be called to ensure that all rows are actually inserted into TB_PEANUT_BRICKALIGN. Failure to do so will result in incomplete writes.
			/// </summary>
			static public void Flush() { if (cmd == null || index == 0) return; cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			private static long [] a_ID_EVENTBRICK = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_EVENTBRICK for the current row.
			/// </summary>
			public long _ID_EVENTBRICK
			{
				get
				{
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			private static long [] a_ID_PROCESSOPERATION = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_PROCESSOPERATION for the current row.
			/// </summary>
			public long _ID_PROCESSOPERATION
			{
				get
				{
					return System.Convert.ToInt64(m_DR[1]);
				}
			}
			private static double [] a_TDX = new double[ArraySize];
			/// <summary>
			/// Retrieves TDX for the current row.
			/// </summary>
			public double _TDX
			{
				get
				{
					return System.Convert.ToDouble(m_DR[2]);
				}
			}
			private static double [] a_TDY = new double[ArraySize];
			/// <summary>
			/// Retrieves TDY for the current row.
			/// </summary>
			public double _TDY
			{
				get
				{
					return System.Convert.ToDouble(m_DR[3]);
				}
			}
			private static double [] a_TDZ = new double[ArraySize];
			/// <summary>
			/// Retrieves TDZ for the current row.
			/// </summary>
			public double _TDZ
			{
				get
				{
					return System.Convert.ToDouble(m_DR[4]);
				}
			}
			private static double [] a_SDX = new double[ArraySize];
			/// <summary>
			/// Retrieves SDX for the current row.
			/// </summary>
			public double _SDX
			{
				get
				{
					return System.Convert.ToDouble(m_DR[5]);
				}
			}
			private static double [] a_SDY = new double[ArraySize];
			/// <summary>
			/// Retrieves SDY for the current row.
			/// </summary>
			public double _SDY
			{
				get
				{
					return System.Convert.ToDouble(m_DR[6]);
				}
			}
			private static double [] a_TXX = new double[ArraySize];
			/// <summary>
			/// Retrieves TXX for the current row.
			/// </summary>
			public double _TXX
			{
				get
				{
					return System.Convert.ToDouble(m_DR[7]);
				}
			}
			private static double [] a_TXY = new double[ArraySize];
			/// <summary>
			/// Retrieves TXY for the current row.
			/// </summary>
			public double _TXY
			{
				get
				{
					return System.Convert.ToDouble(m_DR[8]);
				}
			}
			private static double [] a_TYX = new double[ArraySize];
			/// <summary>
			/// Retrieves TYX for the current row.
			/// </summary>
			public double _TYX
			{
				get
				{
					return System.Convert.ToDouble(m_DR[9]);
				}
			}
			private static double [] a_TYY = new double[ArraySize];
			/// <summary>
			/// Retrieves TYY for the current row.
			/// </summary>
			public double _TYY
			{
				get
				{
					return System.Convert.ToDouble(m_DR[10]);
				}
			}
			private static double [] a_SIGMAPOSX = new double[ArraySize];
			/// <summary>
			/// Retrieves SIGMAPOSX for the current row.
			/// </summary>
			public double _SIGMAPOSX
			{
				get
				{
					return System.Convert.ToDouble(m_DR[11]);
				}
			}
			private static double [] a_SIGMAPOSY = new double[ArraySize];
			/// <summary>
			/// Retrieves SIGMAPOSY for the current row.
			/// </summary>
			public double _SIGMAPOSY
			{
				get
				{
					return System.Convert.ToDouble(m_DR[12]);
				}
			}
			private static double [] a_SIGMASLOPEX = new double[ArraySize];
			/// <summary>
			/// Retrieves SIGMASLOPEX for the current row.
			/// </summary>
			public double _SIGMASLOPEX
			{
				get
				{
					return System.Convert.ToDouble(m_DR[13]);
				}
			}
			private static double [] a_SIGMASLOPEY = new double[ArraySize];
			/// <summary>
			/// Retrieves SIGMASLOPEY for the current row.
			/// </summary>
			public double _SIGMASLOPEY
			{
				get
				{
					return System.Convert.ToDouble(m_DR[14]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is actually inserted later, in a batch insertion command. The Flush method must be called to ensure all rows are actually written.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">the value to be inserted for ID_EVENTBRICK.</param>
			/// <param name="i_ID_PROCESSOPERATION">the value to be inserted for ID_PROCESSOPERATION.</param>
			/// <param name="i_TDX">the value to be inserted for TDX.</param>
			/// <param name="i_TDY">the value to be inserted for TDY.</param>
			/// <param name="i_TDZ">the value to be inserted for TDZ.</param>
			/// <param name="i_SDX">the value to be inserted for SDX.</param>
			/// <param name="i_SDY">the value to be inserted for SDY.</param>
			/// <param name="i_TXX">the value to be inserted for TXX.</param>
			/// <param name="i_TXY">the value to be inserted for TXY.</param>
			/// <param name="i_TYX">the value to be inserted for TYX.</param>
			/// <param name="i_TYY">the value to be inserted for TYY.</param>
			/// <param name="i_SIGMAPOSX">the value to be inserted for SIGMAPOSX.</param>
			/// <param name="i_SIGMAPOSY">the value to be inserted for SIGMAPOSY.</param>
			/// <param name="i_SIGMASLOPEX">the value to be inserted for SIGMASLOPEX.</param>
			/// <param name="i_SIGMASLOPEY">the value to be inserted for SIGMASLOPEY.</param>
			static public void Insert(long i_ID_EVENTBRICK,long i_ID_PROCESSOPERATION,double i_TDX,double i_TDY,double i_TDZ,double i_SDX,double i_SDY,double i_TXX,double i_TXY,double i_TYX,double i_TYY,double i_SIGMAPOSX,double i_SIGMAPOSY,double i_SIGMASLOPEX,double i_SIGMASLOPEY)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				a_ID_EVENTBRICK[index] = i_ID_EVENTBRICK;
				a_ID_PROCESSOPERATION[index] = i_ID_PROCESSOPERATION;
				a_TDX[index] = i_TDX;
				a_TDY[index] = i_TDY;
				a_TDZ[index] = i_TDZ;
				a_SDX[index] = i_SDX;
				a_SDY[index] = i_SDY;
				a_TXX[index] = i_TXX;
				a_TXY[index] = i_TXY;
				a_TYX[index] = i_TYX;
				a_TYY[index] = i_TYY;
				a_SIGMAPOSX[index] = i_SIGMAPOSX;
				a_SIGMAPOSY[index] = i_SIGMAPOSY;
				a_SIGMASLOPEX[index] = i_SIGMASLOPEX;
				a_SIGMASLOPEY[index] = i_SIGMASLOPEY;
				if (++index >= ArraySize) { cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			}
			/// <summary>
			/// Reads a set of rows from TB_PEANUT_BRICKALIGN and retrieves them into a new TB_PEANUT_BRICKALIGN object.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_PROCESSOPERATION">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="order">the ordering scheme to be applied to returned rows. This applies to all columns in the primary key.</param>
			/// <returns>a new instance of the TB_PEANUT_BRICKALIGN class that can be used to read the retrieved data.</returns>
			static public TB_PEANUT_BRICKALIGN SelectPrimaryKey(object i_ID_EVENTBRICK,object i_ID_PROCESSOPERATION, OrderBy order)
			{
				string wherestr = "";
				string wtempstr = "";
				if (i_ID_EVENTBRICK != null)
				{
					if (i_ID_EVENTBRICK == System.DBNull.Value) wtempstr = "ID_EVENTBRICK IS NULL";
					else wtempstr = "ID_EVENTBRICK = " + i_ID_EVENTBRICK.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_PROCESSOPERATION != null)
				{
					if (i_ID_PROCESSOPERATION == System.DBNull.Value) wtempstr = "ID_PROCESSOPERATION IS NULL";
					else wtempstr = "ID_PROCESSOPERATION = " + i_ID_PROCESSOPERATION.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (order == OrderBy.Ascending) return SelectWhere(wherestr, "ID_EVENTBRICK ASC,ID_PROCESSOPERATION ASC");
				else if (order == OrderBy.Descending) return SelectWhere(wherestr, "ID_EVENTBRICK DESC,ID_PROCESSOPERATION DESC");
				return SelectWhere(wherestr, null);
			}
			/// <summary>
			/// Reads a set of rows from TB_PEANUT_BRICKALIGN and retrieves them into a new TB_PEANUT_BRICKALIGN object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the TB_PEANUT_BRICKALIGN class that can be used to read the retrieved data.</returns>
			static public TB_PEANUT_BRICKALIGN SelectWhere(string wherestr, string orderstr)
			{
				TB_PEANUT_BRICKALIGN newobj = new TB_PEANUT_BRICKALIGN();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_EVENTBRICK,ID_PROCESSOPERATION,TDX,TDY,TDZ,SDX,SDY,TXX,TXY,TYX,TYY,SIGMAPOSX,SIGMAPOSY,SIGMASLOPEX,SIGMASLOPEY FROM TB_PEANUT_BRICKALIGN" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO TB_PEANUT_BRICKALIGN (ID_EVENTBRICK,ID_PROCESSOPERATION,TDX,TDY,TDZ,SDX,SDY,TXX,TXY,TYX,TYY,SIGMAPOSX,SIGMAPOSY,SIGMASLOPEX,SIGMASLOPEY) VALUES (:p_1,:p_2,:p_3,:p_4,:p_5,:p_6,:p_7,:p_8,:p_9,:p_10,:p_11,:p_12,:p_13,:p_14,:p_15)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_EVENTBRICK;
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_PROCESSOPERATION;
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_TDX;
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_TDY;
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_TDZ;
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_SDX;
				newcmd.Parameters.Add("p_7", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_SDY;
				newcmd.Parameters.Add("p_8", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_TXX;
				newcmd.Parameters.Add("p_9", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_TXY;
				newcmd.Parameters.Add("p_10", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_TYX;
				newcmd.Parameters.Add("p_11", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_TYY;
				newcmd.Parameters.Add("p_12", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_SIGMAPOSX;
				newcmd.Parameters.Add("p_13", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_SIGMAPOSY;
				newcmd.Parameters.Add("p_14", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_SIGMASLOPEX;
				newcmd.Parameters.Add("p_15", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_SIGMASLOPEY;
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the TB_PEANUT_BRICKINFO table in the DB.
		/// For data insertion, the Insert and Flush static methods are used. This class inserts data in batches.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class TB_PEANUT_BRICKINFO
		{
			internal TB_PEANUT_BRICKINFO() {}
			System.Data.DataRowCollection m_DRC;
			const int ArraySize = 100;
			static int index = 0;
			/// <summary>
			/// Since this class uses batch insertion, this method must be called to ensure that all rows are actually inserted into TB_PEANUT_BRICKINFO. Failure to do so will result in incomplete writes.
			/// </summary>
			static public void Flush() { if (cmd == null || index == 0) return; cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			private static long [] a_ID_EVENTBRICK = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_EVENTBRICK for the current row.
			/// </summary>
			public long _ID_EVENTBRICK
			{
				get
				{
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			private static System.DateTime [] a_EXPOSURESTART = new System.DateTime[ArraySize];
			/// <summary>
			/// Retrieves EXPOSURESTART for the current row.
			/// </summary>
			public System.DateTime _EXPOSURESTART
			{
				get
				{
					return System.Convert.ToDateTime(m_DR[1]);
				}
			}
			private static System.DateTime [] a_EXPOSUREFINISH = new System.DateTime[ArraySize];
			/// <summary>
			/// Retrieves EXPOSUREFINISH for the current row.
			/// </summary>
			public System.DateTime _EXPOSUREFINISH
			{
				get
				{
					return System.Convert.ToDateTime(m_DR[2]);
				}
			}
			private static long [] a_POSITIONCODE = new long[ArraySize];
			/// <summary>
			/// Retrieves POSITIONCODE for the current row.
			/// </summary>
			public long _POSITIONCODE
			{
				get
				{
					return System.Convert.ToInt64(m_DR[3]);
				}
			}
			private static object [] a_COSMICHOURS = new object[ArraySize];
			/// <summary>
			/// Retrieves COSMICHOURS for the current row. The return value can be System.DBNull.Value or a value that can be cast to double.
			/// </summary>
			public object _COSMICHOURS
			{
				get
				{
					if (m_DR[4] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDouble(m_DR[4]);
				}
			}
			private static object [] a_ID_SITE = new object[ArraySize];
			/// <summary>
			/// Retrieves ID_SITE for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _ID_SITE
			{
				get
				{
					if (m_DR[5] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[5]);
				}
			}
			private static string [] a_LABORATORY = new string[ArraySize];
			/// <summary>
			/// Retrieves LABORATORY for the current row.
			/// </summary>
			public string _LABORATORY
			{
				get
				{
					return System.Convert.ToString(m_DR[6]);
				}
			}
			private static object [] a_NOTES = new object[ArraySize];
			/// <summary>
			/// Retrieves NOTES for the current row. The return value can be System.DBNull.Value or a value that can be cast to string.
			/// </summary>
			public object _NOTES
			{
				get
				{
					if (m_DR[7] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToString(m_DR[7]);
				}
			}
			private static object [] a_POT = new object[ArraySize];
			/// <summary>
			/// Retrieves POT for the current row. The return value can be System.DBNull.Value or a value that can be cast to double.
			/// </summary>
			public object _POT
			{
				get
				{
					if (m_DR[8] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDouble(m_DR[8]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is actually inserted later, in a batch insertion command. The Flush method must be called to ensure all rows are actually written.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">the value to be inserted for ID_EVENTBRICK.</param>
			/// <param name="i_EXPOSURESTART">the value to be inserted for EXPOSURESTART.</param>
			/// <param name="i_EXPOSUREFINISH">the value to be inserted for EXPOSUREFINISH.</param>
			/// <param name="i_POSITIONCODE">the value to be inserted for POSITIONCODE.</param>
			/// <param name="i_COSMICHOURS">the value to be inserted for COSMICHOURS. The value for this parameter can be double or System.DBNull.Value.</param>
			/// <param name="i_ID_SITE">the value to be inserted for ID_SITE. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_LABORATORY">the value to be inserted for LABORATORY.</param>
			/// <param name="i_NOTES">the value to be inserted for NOTES. The value for this parameter can be string or System.DBNull.Value.</param>
			/// <param name="i_POT">the value to be inserted for POT. The value for this parameter can be double or System.DBNull.Value.</param>
			static public void Insert(long i_ID_EVENTBRICK,System.DateTime i_EXPOSURESTART,System.DateTime i_EXPOSUREFINISH,long i_POSITIONCODE,object i_COSMICHOURS,object i_ID_SITE,string i_LABORATORY,object i_NOTES,object i_POT)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				a_ID_EVENTBRICK[index] = i_ID_EVENTBRICK;
				a_EXPOSURESTART[index] = i_EXPOSURESTART;
				a_EXPOSUREFINISH[index] = i_EXPOSUREFINISH;
				a_POSITIONCODE[index] = i_POSITIONCODE;
				a_COSMICHOURS[index] = i_COSMICHOURS;
				a_ID_SITE[index] = i_ID_SITE;
				a_LABORATORY[index] = i_LABORATORY;
				a_NOTES[index] = i_NOTES;
				a_POT[index] = i_POT;
				if (++index >= ArraySize) { cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			}
			/// <summary>
			/// Reads a set of rows from TB_PEANUT_BRICKINFO and retrieves them into a new TB_PEANUT_BRICKINFO object.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="order">the ordering scheme to be applied to returned rows.</param>
			/// <returns>a new instance of the TB_PEANUT_BRICKINFO class that can be used to read the retrieved data.</returns>
			static public TB_PEANUT_BRICKINFO SelectPrimaryKey(object i_ID_EVENTBRICK, OrderBy order)
			{
				string wherestr = "";
				string wtempstr = "";
				if (i_ID_EVENTBRICK != null)
				{
					if (i_ID_EVENTBRICK == System.DBNull.Value) wtempstr = "ID_EVENTBRICK IS NULL";
					else wtempstr = "ID_EVENTBRICK = " + i_ID_EVENTBRICK.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (order == OrderBy.Ascending) return SelectWhere(wherestr, "ID_EVENTBRICK ASC");
				else if (order == OrderBy.Descending) return SelectWhere(wherestr, "ID_EVENTBRICK DESC");
				return SelectWhere(wherestr, null);
			}
			/// <summary>
			/// Reads a set of rows from TB_PEANUT_BRICKINFO and retrieves them into a new TB_PEANUT_BRICKINFO object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the TB_PEANUT_BRICKINFO class that can be used to read the retrieved data.</returns>
			static public TB_PEANUT_BRICKINFO SelectWhere(string wherestr, string orderstr)
			{
				TB_PEANUT_BRICKINFO newobj = new TB_PEANUT_BRICKINFO();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_EVENTBRICK,EXPOSURESTART,EXPOSUREFINISH,POSITIONCODE,COSMICHOURS,ID_SITE,LABORATORY,NOTES,POT FROM TB_PEANUT_BRICKINFO" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO TB_PEANUT_BRICKINFO (ID_EVENTBRICK,EXPOSURESTART,EXPOSUREFINISH,POSITIONCODE,COSMICHOURS,ID_SITE,LABORATORY,NOTES,POT) VALUES (:p_1,:p_2,:p_3,:p_4,:p_5,:p_6,:p_7,:p_8,:p_9)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_EVENTBRICK;
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.DateTime, System.Data.ParameterDirection.Input).Value = a_EXPOSURESTART;
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.DateTime, System.Data.ParameterDirection.Input).Value = a_EXPOSUREFINISH;
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_POSITIONCODE;
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_COSMICHOURS;
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_SITE;
				newcmd.Parameters.Add("p_7", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input).Value = a_LABORATORY;
				newcmd.Parameters.Add("p_8", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input).Value = a_NOTES;
				newcmd.Parameters.Add("p_9", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_POT;
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the TB_PEANUT_HITS table in the DB.
		/// For data insertion, the Insert and Flush static methods are used. This class inserts data in batches.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class TB_PEANUT_HITS
		{
			internal TB_PEANUT_HITS() {}
			System.Data.DataRowCollection m_DRC;
			const int ArraySize = 100;
			static int index = 0;
			/// <summary>
			/// Since this class uses batch insertion, this method must be called to ensure that all rows are actually inserted into TB_PEANUT_HITS. Failure to do so will result in incomplete writes.
			/// </summary>
			static public void Flush() { if (cmd == null || index == 0) return; cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			private static long [] a_ID_EVENT = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_EVENT for the current row.
			/// </summary>
			public long _ID_EVENT
			{
				get
				{
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			private static int [] a_PLANE_ID = new int[ArraySize];
			/// <summary>
			/// Retrieves PLANE_ID for the current row.
			/// </summary>
			public int _PLANE_ID
			{
				get
				{
					return System.Convert.ToInt32(m_DR[1]);
				}
			}
			private static long [] a_HIT_ID = new long[ArraySize];
			/// <summary>
			/// Retrieves HIT_ID for the current row.
			/// </summary>
			public long _HIT_ID
			{
				get
				{
					return System.Convert.ToInt64(m_DR[2]);
				}
			}
			private static double [] a_TCOORD = new double[ArraySize];
			/// <summary>
			/// Retrieves TCOORD for the current row.
			/// </summary>
			public double _TCOORD
			{
				get
				{
					return System.Convert.ToDouble(m_DR[3]);
				}
			}
			private static double [] a_Z = new double[ArraySize];
			/// <summary>
			/// Retrieves Z for the current row.
			/// </summary>
			public double _Z
			{
				get
				{
					return System.Convert.ToDouble(m_DR[4]);
				}
			}
			private static double [] a_BRIGHTNESS = new double[ArraySize];
			/// <summary>
			/// Retrieves BRIGHTNESS for the current row.
			/// </summary>
			public double _BRIGHTNESS
			{
				get
				{
					return System.Convert.ToDouble(m_DR[5]);
				}
			}
			private static double [] a_CCDX = new double[ArraySize];
			/// <summary>
			/// Retrieves CCDX for the current row.
			/// </summary>
			public double _CCDX
			{
				get
				{
					return System.Convert.ToDouble(m_DR[6]);
				}
			}
			private static double [] a_CCDY = new double[ArraySize];
			/// <summary>
			/// Retrieves CCDY for the current row.
			/// </summary>
			public double _CCDY
			{
				get
				{
					return System.Convert.ToDouble(m_DR[7]);
				}
			}
			private static char [] a_PROJ_ID = new char[ArraySize];
			/// <summary>
			/// Retrieves PROJ_ID for the current row.
			/// </summary>
			public char _PROJ_ID
			{
				get
				{
					return System.Convert.ToChar(m_DR[8]);
				}
			}
			private static char [] a_II_CHAIN_ID = new char[ArraySize];
			/// <summary>
			/// Retrieves II_CHAIN_ID for the current row.
			/// </summary>
			public char _II_CHAIN_ID
			{
				get
				{
					return System.Convert.ToChar(m_DR[9]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is actually inserted later, in a batch insertion command. The Flush method must be called to ensure all rows are actually written.
			/// </summary>
			/// <param name="i_ID_EVENT">the value to be inserted for ID_EVENT.</param>
			/// <param name="i_PLANE_ID">the value to be inserted for PLANE_ID.</param>
			/// <param name="i_HIT_ID">the value to be inserted for HIT_ID.</param>
			/// <param name="i_TCOORD">the value to be inserted for TCOORD.</param>
			/// <param name="i_Z">the value to be inserted for Z.</param>
			/// <param name="i_BRIGHTNESS">the value to be inserted for BRIGHTNESS.</param>
			/// <param name="i_CCDX">the value to be inserted for CCDX.</param>
			/// <param name="i_CCDY">the value to be inserted for CCDY.</param>
			/// <param name="i_PROJ_ID">the value to be inserted for PROJ_ID.</param>
			/// <param name="i_II_CHAIN_ID">the value to be inserted for II_CHAIN_ID.</param>
			static public void Insert(long i_ID_EVENT,int i_PLANE_ID,long i_HIT_ID,double i_TCOORD,double i_Z,double i_BRIGHTNESS,double i_CCDX,double i_CCDY,char i_PROJ_ID,char i_II_CHAIN_ID)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				a_ID_EVENT[index] = i_ID_EVENT;
				a_PLANE_ID[index] = i_PLANE_ID;
				a_HIT_ID[index] = i_HIT_ID;
				a_TCOORD[index] = i_TCOORD;
				a_Z[index] = i_Z;
				a_BRIGHTNESS[index] = i_BRIGHTNESS;
				a_CCDX[index] = i_CCDX;
				a_CCDY[index] = i_CCDY;
				a_PROJ_ID[index] = i_PROJ_ID;
				a_II_CHAIN_ID[index] = i_II_CHAIN_ID;
				if (++index >= ArraySize) { cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			}
			/// <summary>
			/// Reads a set of rows from TB_PEANUT_HITS and retrieves them into a new TB_PEANUT_HITS object.
			/// </summary>
			/// <param name="i_ID_EVENT">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_PLANE_ID">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_HIT_ID">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_PROJ_ID">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="order">the ordering scheme to be applied to returned rows. This applies to all columns in the primary key.</param>
			/// <returns>a new instance of the TB_PEANUT_HITS class that can be used to read the retrieved data.</returns>
			static public TB_PEANUT_HITS SelectPrimaryKey(object i_ID_EVENT,object i_PLANE_ID,object i_HIT_ID,object i_PROJ_ID, OrderBy order)
			{
				string wherestr = "";
				string wtempstr = "";
				if (i_ID_EVENT != null)
				{
					if (i_ID_EVENT == System.DBNull.Value) wtempstr = "ID_EVENT IS NULL";
					else wtempstr = "ID_EVENT = " + i_ID_EVENT.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_PLANE_ID != null)
				{
					if (i_PLANE_ID == System.DBNull.Value) wtempstr = "PLANE_ID IS NULL";
					else wtempstr = "PLANE_ID = " + i_PLANE_ID.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_HIT_ID != null)
				{
					if (i_HIT_ID == System.DBNull.Value) wtempstr = "HIT_ID IS NULL";
					else wtempstr = "HIT_ID = " + i_HIT_ID.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_PROJ_ID != null)
				{
					if (i_PROJ_ID == System.DBNull.Value) wtempstr = "PROJ_ID IS NULL";
					else wtempstr = "PROJ_ID = " + i_PROJ_ID.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (order == OrderBy.Ascending) return SelectWhere(wherestr, "ID_EVENT ASC,PLANE_ID ASC,HIT_ID ASC,PROJ_ID ASC");
				else if (order == OrderBy.Descending) return SelectWhere(wherestr, "ID_EVENT DESC,PLANE_ID DESC,HIT_ID DESC,PROJ_ID DESC");
				return SelectWhere(wherestr, null);
			}
			/// <summary>
			/// Reads a set of rows from TB_PEANUT_HITS and retrieves them into a new TB_PEANUT_HITS object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the TB_PEANUT_HITS class that can be used to read the retrieved data.</returns>
			static public TB_PEANUT_HITS SelectWhere(string wherestr, string orderstr)
			{
				TB_PEANUT_HITS newobj = new TB_PEANUT_HITS();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_EVENT,PLANE_ID,HIT_ID,TCOORD,Z,BRIGHTNESS,CCDX,CCDY,PROJ_ID,II_CHAIN_ID FROM TB_PEANUT_HITS" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO TB_PEANUT_HITS (ID_EVENT,PLANE_ID,HIT_ID,TCOORD,Z,BRIGHTNESS,CCDX,CCDY,PROJ_ID,II_CHAIN_ID) VALUES (:p_1,:p_2,:p_3,:p_4,:p_5,:p_6,:p_7,:p_8,:p_9,:p_10)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_EVENT;
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Int, System.Data.ParameterDirection.Input).Value = a_PLANE_ID;
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_HIT_ID;
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_TCOORD;
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_Z;
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_BRIGHTNESS;
				newcmd.Parameters.Add("p_7", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_CCDX;
				newcmd.Parameters.Add("p_8", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_CCDY;
				newcmd.Parameters.Add("p_9", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input).Value = a_PROJ_ID;
				newcmd.Parameters.Add("p_10", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input).Value = a_II_CHAIN_ID;
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the TB_PEANUT_PREDTRACKBRICKS table in the DB.
		/// For data insertion, the Insert and Flush static methods are used. This class inserts data in batches.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class TB_PEANUT_PREDTRACKBRICKS
		{
			internal TB_PEANUT_PREDTRACKBRICKS() {}
			System.Data.DataRowCollection m_DRC;
			const int ArraySize = 100;
			static int index = 0;
			/// <summary>
			/// Since this class uses batch insertion, this method must be called to ensure that all rows are actually inserted into TB_PEANUT_PREDTRACKBRICKS. Failure to do so will result in incomplete writes.
			/// </summary>
			static public void Flush() { if (cmd == null || index == 0) return; cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			private static long [] a_ID_PROCESSOPERATION = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_PROCESSOPERATION for the current row.
			/// </summary>
			public long _ID_PROCESSOPERATION
			{
				get
				{
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			private static long [] a_ID_EVENTBRICK = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_EVENTBRICK for the current row.
			/// </summary>
			public long _ID_EVENTBRICK
			{
				get
				{
					return System.Convert.ToInt64(m_DR[1]);
				}
			}
			private static long [] a_ID_EVENT = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_EVENT for the current row.
			/// </summary>
			public long _ID_EVENT
			{
				get
				{
					return System.Convert.ToInt64(m_DR[2]);
				}
			}
			private static long [] a_TRACK = new long[ArraySize];
			/// <summary>
			/// Retrieves TRACK for the current row.
			/// </summary>
			public long _TRACK
			{
				get
				{
					return System.Convert.ToInt64(m_DR[3]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is actually inserted later, in a batch insertion command. The Flush method must be called to ensure all rows are actually written.
			/// </summary>
			/// <param name="i_ID_PROCESSOPERATION">the value to be inserted for ID_PROCESSOPERATION.</param>
			/// <param name="i_ID_EVENTBRICK">the value to be inserted for ID_EVENTBRICK.</param>
			/// <param name="i_ID_EVENT">the value to be inserted for ID_EVENT.</param>
			/// <param name="i_TRACK">the value to be inserted for TRACK.</param>
			static public void Insert(long i_ID_PROCESSOPERATION,long i_ID_EVENTBRICK,long i_ID_EVENT,long i_TRACK)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				a_ID_PROCESSOPERATION[index] = i_ID_PROCESSOPERATION;
				a_ID_EVENTBRICK[index] = i_ID_EVENTBRICK;
				a_ID_EVENT[index] = i_ID_EVENT;
				a_TRACK[index] = i_TRACK;
				if (++index >= ArraySize) { cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			}
			/// <summary>
			/// Reads a set of rows from TB_PEANUT_PREDTRACKBRICKS and retrieves them into a new TB_PEANUT_PREDTRACKBRICKS object.
			/// </summary>
			/// <param name="i_ID_PROCESSOPERATION">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_EVENTBRICK">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_EVENT">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_TRACK">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="order">the ordering scheme to be applied to returned rows. This applies to all columns in the primary key.</param>
			/// <returns>a new instance of the TB_PEANUT_PREDTRACKBRICKS class that can be used to read the retrieved data.</returns>
			static public TB_PEANUT_PREDTRACKBRICKS SelectPrimaryKey(object i_ID_PROCESSOPERATION,object i_ID_EVENTBRICK,object i_ID_EVENT,object i_TRACK, OrderBy order)
			{
				string wherestr = "";
				string wtempstr = "";
				if (i_ID_PROCESSOPERATION != null)
				{
					if (i_ID_PROCESSOPERATION == System.DBNull.Value) wtempstr = "ID_PROCESSOPERATION IS NULL";
					else wtempstr = "ID_PROCESSOPERATION = " + i_ID_PROCESSOPERATION.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_EVENTBRICK != null)
				{
					if (i_ID_EVENTBRICK == System.DBNull.Value) wtempstr = "ID_EVENTBRICK IS NULL";
					else wtempstr = "ID_EVENTBRICK = " + i_ID_EVENTBRICK.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_EVENT != null)
				{
					if (i_ID_EVENT == System.DBNull.Value) wtempstr = "ID_EVENT IS NULL";
					else wtempstr = "ID_EVENT = " + i_ID_EVENT.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_TRACK != null)
				{
					if (i_TRACK == System.DBNull.Value) wtempstr = "TRACK IS NULL";
					else wtempstr = "TRACK = " + i_TRACK.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (order == OrderBy.Ascending) return SelectWhere(wherestr, "ID_PROCESSOPERATION ASC,ID_EVENTBRICK ASC,ID_EVENT ASC,TRACK ASC");
				else if (order == OrderBy.Descending) return SelectWhere(wherestr, "ID_PROCESSOPERATION DESC,ID_EVENTBRICK DESC,ID_EVENT DESC,TRACK DESC");
				return SelectWhere(wherestr, null);
			}
			/// <summary>
			/// Reads a set of rows from TB_PEANUT_PREDTRACKBRICKS and retrieves them into a new TB_PEANUT_PREDTRACKBRICKS object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the TB_PEANUT_PREDTRACKBRICKS class that can be used to read the retrieved data.</returns>
			static public TB_PEANUT_PREDTRACKBRICKS SelectWhere(string wherestr, string orderstr)
			{
				TB_PEANUT_PREDTRACKBRICKS newobj = new TB_PEANUT_PREDTRACKBRICKS();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_PROCESSOPERATION,ID_EVENTBRICK,ID_EVENT,TRACK FROM TB_PEANUT_PREDTRACKBRICKS" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO TB_PEANUT_PREDTRACKBRICKS (ID_PROCESSOPERATION,ID_EVENTBRICK,ID_EVENT,TRACK) VALUES (:p_1,:p_2,:p_3,:p_4)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_PROCESSOPERATION;
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_EVENTBRICK;
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_EVENT;
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_TRACK;
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the TB_PEANUT_PREDTRACKS table in the DB.
		/// For data insertion, the Insert and Flush static methods are used. This class inserts data in batches.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class TB_PEANUT_PREDTRACKS
		{
			internal TB_PEANUT_PREDTRACKS() {}
			System.Data.DataRowCollection m_DRC;
			const int ArraySize = 100;
			static int index = 0;
			/// <summary>
			/// Since this class uses batch insertion, this method must be called to ensure that all rows are actually inserted into TB_PEANUT_PREDTRACKS. Failure to do so will result in incomplete writes.
			/// </summary>
			static public void Flush() { if (cmd == null || index == 0) return; cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			private static long [] a_ID_EVENT = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_EVENT for the current row.
			/// </summary>
			public long _ID_EVENT
			{
				get
				{
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			private static char [] a_PROJ_ID = new char[ArraySize];
			/// <summary>
			/// Retrieves PROJ_ID for the current row.
			/// </summary>
			public char _PROJ_ID
			{
				get
				{
					return System.Convert.ToChar(m_DR[1]);
				}
			}
			private static long [] a_TRACK_ID = new long[ArraySize];
			/// <summary>
			/// Retrieves TRACK_ID for the current row.
			/// </summary>
			public long _TRACK_ID
			{
				get
				{
					return System.Convert.ToInt64(m_DR[2]);
				}
			}
			private static object [] a_TRK_ID_X = new object[ArraySize];
			/// <summary>
			/// Retrieves TRK_ID_X for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _TRK_ID_X
			{
				get
				{
					if (m_DR[3] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[3]);
				}
			}
			private static object [] a_TRK_ID_Y = new object[ArraySize];
			/// <summary>
			/// Retrieves TRK_ID_Y for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _TRK_ID_Y
			{
				get
				{
					if (m_DR[4] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[4]);
				}
			}
			private static double [] a_ACOORD = new double[ArraySize];
			/// <summary>
			/// Retrieves ACOORD for the current row.
			/// </summary>
			public double _ACOORD
			{
				get
				{
					return System.Convert.ToDouble(m_DR[5]);
				}
			}
			private static double [] a_BCOORD = new double[ArraySize];
			/// <summary>
			/// Retrieves BCOORD for the current row.
			/// </summary>
			public double _BCOORD
			{
				get
				{
					return System.Convert.ToDouble(m_DR[6]);
				}
			}
			private static object [] a_NHITS = new object[ArraySize];
			/// <summary>
			/// Retrieves NHITS for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _NHITS
			{
				get
				{
					if (m_DR[7] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[7]);
				}
			}
			private static char [] a_PLANE_X = new char[ArraySize];
			/// <summary>
			/// Retrieves PLANE_X for the current row.
			/// </summary>
			public char _PLANE_X
			{
				get
				{
					return System.Convert.ToChar(m_DR[8]);
				}
			}
			private static char [] a_PLANE_Y = new char[ArraySize];
			/// <summary>
			/// Retrieves PLANE_Y for the current row.
			/// </summary>
			public char _PLANE_Y
			{
				get
				{
					return System.Convert.ToChar(m_DR[9]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is actually inserted later, in a batch insertion command. The Flush method must be called to ensure all rows are actually written.
			/// </summary>
			/// <param name="i_ID_EVENT">the value to be inserted for ID_EVENT.</param>
			/// <param name="i_PROJ_ID">the value to be inserted for PROJ_ID.</param>
			/// <param name="i_TRACK_ID">the value to be inserted for TRACK_ID.</param>
			/// <param name="i_TRK_ID_X">the value to be inserted for TRK_ID_X. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_TRK_ID_Y">the value to be inserted for TRK_ID_Y. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_ACOORD">the value to be inserted for ACOORD.</param>
			/// <param name="i_BCOORD">the value to be inserted for BCOORD.</param>
			/// <param name="i_NHITS">the value to be inserted for NHITS. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_PLANE_X">the value to be inserted for PLANE_X.</param>
			/// <param name="i_PLANE_Y">the value to be inserted for PLANE_Y.</param>
			static public void Insert(long i_ID_EVENT,char i_PROJ_ID,long i_TRACK_ID,object i_TRK_ID_X,object i_TRK_ID_Y,double i_ACOORD,double i_BCOORD,object i_NHITS,char i_PLANE_X,char i_PLANE_Y)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				a_ID_EVENT[index] = i_ID_EVENT;
				a_PROJ_ID[index] = i_PROJ_ID;
				a_TRACK_ID[index] = i_TRACK_ID;
				a_TRK_ID_X[index] = i_TRK_ID_X;
				a_TRK_ID_Y[index] = i_TRK_ID_Y;
				a_ACOORD[index] = i_ACOORD;
				a_BCOORD[index] = i_BCOORD;
				a_NHITS[index] = i_NHITS;
				a_PLANE_X[index] = i_PLANE_X;
				a_PLANE_Y[index] = i_PLANE_Y;
				if (++index >= ArraySize) { cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			}
			/// <summary>
			/// Reads a set of rows from TB_PEANUT_PREDTRACKS and retrieves them into a new TB_PEANUT_PREDTRACKS object.
			/// </summary>
			/// <param name="i_ID_EVENT">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_PROJ_ID">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_TRACK_ID">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="order">the ordering scheme to be applied to returned rows. This applies to all columns in the primary key.</param>
			/// <returns>a new instance of the TB_PEANUT_PREDTRACKS class that can be used to read the retrieved data.</returns>
			static public TB_PEANUT_PREDTRACKS SelectPrimaryKey(object i_ID_EVENT,object i_PROJ_ID,object i_TRACK_ID, OrderBy order)
			{
				string wherestr = "";
				string wtempstr = "";
				if (i_ID_EVENT != null)
				{
					if (i_ID_EVENT == System.DBNull.Value) wtempstr = "ID_EVENT IS NULL";
					else wtempstr = "ID_EVENT = " + i_ID_EVENT.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_PROJ_ID != null)
				{
					if (i_PROJ_ID == System.DBNull.Value) wtempstr = "PROJ_ID IS NULL";
					else wtempstr = "PROJ_ID = " + i_PROJ_ID.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_TRACK_ID != null)
				{
					if (i_TRACK_ID == System.DBNull.Value) wtempstr = "TRACK_ID IS NULL";
					else wtempstr = "TRACK_ID = " + i_TRACK_ID.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (order == OrderBy.Ascending) return SelectWhere(wherestr, "ID_EVENT ASC,PROJ_ID ASC,TRACK_ID ASC");
				else if (order == OrderBy.Descending) return SelectWhere(wherestr, "ID_EVENT DESC,PROJ_ID DESC,TRACK_ID DESC");
				return SelectWhere(wherestr, null);
			}
			/// <summary>
			/// Reads a set of rows from TB_PEANUT_PREDTRACKS and retrieves them into a new TB_PEANUT_PREDTRACKS object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the TB_PEANUT_PREDTRACKS class that can be used to read the retrieved data.</returns>
			static public TB_PEANUT_PREDTRACKS SelectWhere(string wherestr, string orderstr)
			{
				TB_PEANUT_PREDTRACKS newobj = new TB_PEANUT_PREDTRACKS();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_EVENT,PROJ_ID,TRACK_ID,TRK_ID_X,TRK_ID_Y,ACOORD,BCOORD,NHITS,PLANE_X,PLANE_Y FROM TB_PEANUT_PREDTRACKS" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO TB_PEANUT_PREDTRACKS (ID_EVENT,PROJ_ID,TRACK_ID,TRK_ID_X,TRK_ID_Y,ACOORD,BCOORD,NHITS,PLANE_X,PLANE_Y) VALUES (:p_1,:p_2,:p_3,:p_4,:p_5,:p_6,:p_7,:p_8,:p_9,:p_10)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_EVENT;
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input).Value = a_PROJ_ID;
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_TRACK_ID;
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_TRK_ID_X;
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_TRK_ID_Y;
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_ACOORD;
				newcmd.Parameters.Add("p_7", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_BCOORD;
				newcmd.Parameters.Add("p_8", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_NHITS;
				newcmd.Parameters.Add("p_9", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input).Value = a_PLANE_X;
				newcmd.Parameters.Add("p_10", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input).Value = a_PLANE_Y;
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the TB_PEANUT_TRACKHITLINKS table in the DB.
		/// For data insertion, the Insert and Flush static methods are used. This class inserts data in batches.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class TB_PEANUT_TRACKHITLINKS
		{
			internal TB_PEANUT_TRACKHITLINKS() {}
			System.Data.DataRowCollection m_DRC;
			const int ArraySize = 100;
			static int index = 0;
			/// <summary>
			/// Since this class uses batch insertion, this method must be called to ensure that all rows are actually inserted into TB_PEANUT_TRACKHITLINKS. Failure to do so will result in incomplete writes.
			/// </summary>
			static public void Flush() { if (cmd == null || index == 0) return; cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			private static long [] a_ID_EVENT = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_EVENT for the current row.
			/// </summary>
			public long _ID_EVENT
			{
				get
				{
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			private static char [] a_PROJ_ID = new char[ArraySize];
			/// <summary>
			/// Retrieves PROJ_ID for the current row.
			/// </summary>
			public char _PROJ_ID
			{
				get
				{
					return System.Convert.ToChar(m_DR[1]);
				}
			}
			private static long [] a_TRACK_ID = new long[ArraySize];
			/// <summary>
			/// Retrieves TRACK_ID for the current row.
			/// </summary>
			public long _TRACK_ID
			{
				get
				{
					return System.Convert.ToInt64(m_DR[2]);
				}
			}
			private static long [] a_PLANE_ID = new long[ArraySize];
			/// <summary>
			/// Retrieves PLANE_ID for the current row.
			/// </summary>
			public long _PLANE_ID
			{
				get
				{
					return System.Convert.ToInt64(m_DR[3]);
				}
			}
			private static long [] a_HIT_ID = new long[ArraySize];
			/// <summary>
			/// Retrieves HIT_ID for the current row.
			/// </summary>
			public long _HIT_ID
			{
				get
				{
					return System.Convert.ToInt64(m_DR[4]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is actually inserted later, in a batch insertion command. The Flush method must be called to ensure all rows are actually written.
			/// </summary>
			/// <param name="i_ID_EVENT">the value to be inserted for ID_EVENT.</param>
			/// <param name="i_PROJ_ID">the value to be inserted for PROJ_ID.</param>
			/// <param name="i_TRACK_ID">the value to be inserted for TRACK_ID.</param>
			/// <param name="i_PLANE_ID">the value to be inserted for PLANE_ID.</param>
			/// <param name="i_HIT_ID">the value to be inserted for HIT_ID.</param>
			static public void Insert(long i_ID_EVENT,char i_PROJ_ID,long i_TRACK_ID,long i_PLANE_ID,long i_HIT_ID)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				a_ID_EVENT[index] = i_ID_EVENT;
				a_PROJ_ID[index] = i_PROJ_ID;
				a_TRACK_ID[index] = i_TRACK_ID;
				a_PLANE_ID[index] = i_PLANE_ID;
				a_HIT_ID[index] = i_HIT_ID;
				if (++index >= ArraySize) { cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			}
			/// <summary>
			/// Reads a set of rows from TB_PEANUT_TRACKHITLINKS and retrieves them into a new TB_PEANUT_TRACKHITLINKS object.
			/// </summary>
			/// <param name="i_ID_EVENT">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_PROJ_ID">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_TRACK_ID">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_PLANE_ID">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_HIT_ID">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="order">the ordering scheme to be applied to returned rows. This applies to all columns in the primary key.</param>
			/// <returns>a new instance of the TB_PEANUT_TRACKHITLINKS class that can be used to read the retrieved data.</returns>
			static public TB_PEANUT_TRACKHITLINKS SelectPrimaryKey(object i_ID_EVENT,object i_PROJ_ID,object i_TRACK_ID,object i_PLANE_ID,object i_HIT_ID, OrderBy order)
			{
				string wherestr = "";
				string wtempstr = "";
				if (i_ID_EVENT != null)
				{
					if (i_ID_EVENT == System.DBNull.Value) wtempstr = "ID_EVENT IS NULL";
					else wtempstr = "ID_EVENT = " + i_ID_EVENT.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_PROJ_ID != null)
				{
					if (i_PROJ_ID == System.DBNull.Value) wtempstr = "PROJ_ID IS NULL";
					else wtempstr = "PROJ_ID = " + i_PROJ_ID.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_TRACK_ID != null)
				{
					if (i_TRACK_ID == System.DBNull.Value) wtempstr = "TRACK_ID IS NULL";
					else wtempstr = "TRACK_ID = " + i_TRACK_ID.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_PLANE_ID != null)
				{
					if (i_PLANE_ID == System.DBNull.Value) wtempstr = "PLANE_ID IS NULL";
					else wtempstr = "PLANE_ID = " + i_PLANE_ID.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_HIT_ID != null)
				{
					if (i_HIT_ID == System.DBNull.Value) wtempstr = "HIT_ID IS NULL";
					else wtempstr = "HIT_ID = " + i_HIT_ID.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (order == OrderBy.Ascending) return SelectWhere(wherestr, "ID_EVENT ASC,PROJ_ID ASC,TRACK_ID ASC,PLANE_ID ASC,HIT_ID ASC");
				else if (order == OrderBy.Descending) return SelectWhere(wherestr, "ID_EVENT DESC,PROJ_ID DESC,TRACK_ID DESC,PLANE_ID DESC,HIT_ID DESC");
				return SelectWhere(wherestr, null);
			}
			/// <summary>
			/// Reads a set of rows from TB_PEANUT_TRACKHITLINKS and retrieves them into a new TB_PEANUT_TRACKHITLINKS object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the TB_PEANUT_TRACKHITLINKS class that can be used to read the retrieved data.</returns>
			static public TB_PEANUT_TRACKHITLINKS SelectWhere(string wherestr, string orderstr)
			{
				TB_PEANUT_TRACKHITLINKS newobj = new TB_PEANUT_TRACKHITLINKS();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_EVENT,PROJ_ID,TRACK_ID,PLANE_ID,HIT_ID FROM TB_PEANUT_TRACKHITLINKS" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO TB_PEANUT_TRACKHITLINKS (ID_EVENT,PROJ_ID,TRACK_ID,PLANE_ID,HIT_ID) VALUES (:p_1,:p_2,:p_3,:p_4,:p_5)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_EVENT;
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input).Value = a_PROJ_ID;
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_TRACK_ID;
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_PLANE_ID;
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_HIT_ID;
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the TB_PLATES table in the DB.
		/// For data insertion, the Insert and Flush static methods are used. This class inserts data in batches.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class TB_PLATES
		{
			internal TB_PLATES() {}
			System.Data.DataRowCollection m_DRC;
			const int ArraySize = 100;
			static int index = 0;
			/// <summary>
			/// Since this class uses batch insertion, this method must be called to ensure that all rows are actually inserted into TB_PLATES. Failure to do so will result in incomplete writes.
			/// </summary>
			static public void Flush() { if (cmd == null || index == 0) return; cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			private static long [] a_ID_EVENTBRICK = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_EVENTBRICK for the current row.
			/// </summary>
			public long _ID_EVENTBRICK
			{
				get
				{
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			private static long [] a_ID = new long[ArraySize];
			/// <summary>
			/// Retrieves ID for the current row.
			/// </summary>
			public long _ID
			{
				get
				{
					return System.Convert.ToInt64(m_DR[1]);
				}
			}
			private static double [] a_Z = new double[ArraySize];
			/// <summary>
			/// Retrieves Z for the current row.
			/// </summary>
			public double _Z
			{
				get
				{
					return System.Convert.ToDouble(m_DR[2]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is actually inserted later, in a batch insertion command. The Flush method must be called to ensure all rows are actually written.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">the value to be inserted for ID_EVENTBRICK.</param>
			/// <param name="i_ID">the value to be inserted for ID.</param>
			/// <param name="i_Z">the value to be inserted for Z.</param>
			static public void Insert(long i_ID_EVENTBRICK,long i_ID,double i_Z)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				a_ID_EVENTBRICK[index] = i_ID_EVENTBRICK;
				a_ID[index] = i_ID;
				a_Z[index] = i_Z;
				if (++index >= ArraySize) { cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			}
			/// <summary>
			/// Reads a set of rows from TB_PLATES and retrieves them into a new TB_PLATES object.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="order">the ordering scheme to be applied to returned rows. This applies to all columns in the primary key.</param>
			/// <returns>a new instance of the TB_PLATES class that can be used to read the retrieved data.</returns>
			static public TB_PLATES SelectPrimaryKey(object i_ID_EVENTBRICK,object i_ID, OrderBy order)
			{
				string wherestr = "";
				string wtempstr = "";
				if (i_ID_EVENTBRICK != null)
				{
					if (i_ID_EVENTBRICK == System.DBNull.Value) wtempstr = "ID_EVENTBRICK IS NULL";
					else wtempstr = "ID_EVENTBRICK = " + i_ID_EVENTBRICK.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID != null)
				{
					if (i_ID == System.DBNull.Value) wtempstr = "ID IS NULL";
					else wtempstr = "ID = " + i_ID.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (order == OrderBy.Ascending) return SelectWhere(wherestr, "ID_EVENTBRICK ASC,ID ASC");
				else if (order == OrderBy.Descending) return SelectWhere(wherestr, "ID_EVENTBRICK DESC,ID DESC");
				return SelectWhere(wherestr, null);
			}
			/// <summary>
			/// Reads a set of rows from TB_PLATES and retrieves them into a new TB_PLATES object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the TB_PLATES class that can be used to read the retrieved data.</returns>
			static public TB_PLATES SelectWhere(string wherestr, string orderstr)
			{
				TB_PLATES newobj = new TB_PLATES();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_EVENTBRICK,ID,Z FROM TB_PLATES" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO TB_PLATES (ID_EVENTBRICK,ID,Z) VALUES (:p_1,:p_2,:p_3)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_EVENTBRICK;
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID;
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_Z;
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the TB_PLATE_CALIBRATIONS table in the DB.
		/// For data insertion, the Insert and Flush static methods are used. This class inserts data in batches.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class TB_PLATE_CALIBRATIONS
		{
			internal TB_PLATE_CALIBRATIONS() {}
			System.Data.DataRowCollection m_DRC;
			const int ArraySize = 100;
			static int index = 0;
			/// <summary>
			/// Since this class uses batch insertion, this method must be called to ensure that all rows are actually inserted into TB_PLATE_CALIBRATIONS. Failure to do so will result in incomplete writes.
			/// </summary>
			static public void Flush() { if (cmd == null || index == 0) return; cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			private static long [] a_ID_EVENTBRICK = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_EVENTBRICK for the current row.
			/// </summary>
			public long _ID_EVENTBRICK
			{
				get
				{
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			private static long [] a_ID_PLATE = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_PLATE for the current row.
			/// </summary>
			public long _ID_PLATE
			{
				get
				{
					return System.Convert.ToInt64(m_DR[1]);
				}
			}
			private static long [] a_ID_PROCESSOPERATION = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_PROCESSOPERATION for the current row.
			/// </summary>
			public long _ID_PROCESSOPERATION
			{
				get
				{
					return System.Convert.ToInt64(m_DR[2]);
				}
			}
			private static double [] a_Z = new double[ArraySize];
			/// <summary>
			/// Retrieves Z for the current row.
			/// </summary>
			public double _Z
			{
				get
				{
					return System.Convert.ToDouble(m_DR[3]);
				}
			}
			private static double [] a_MAPXX = new double[ArraySize];
			/// <summary>
			/// Retrieves MAPXX for the current row.
			/// </summary>
			public double _MAPXX
			{
				get
				{
					return System.Convert.ToDouble(m_DR[4]);
				}
			}
			private static double [] a_MAPXY = new double[ArraySize];
			/// <summary>
			/// Retrieves MAPXY for the current row.
			/// </summary>
			public double _MAPXY
			{
				get
				{
					return System.Convert.ToDouble(m_DR[5]);
				}
			}
			private static double [] a_MAPYX = new double[ArraySize];
			/// <summary>
			/// Retrieves MAPYX for the current row.
			/// </summary>
			public double _MAPYX
			{
				get
				{
					return System.Convert.ToDouble(m_DR[6]);
				}
			}
			private static double [] a_MAPYY = new double[ArraySize];
			/// <summary>
			/// Retrieves MAPYY for the current row.
			/// </summary>
			public double _MAPYY
			{
				get
				{
					return System.Convert.ToDouble(m_DR[7]);
				}
			}
			private static double [] a_MAPDX = new double[ArraySize];
			/// <summary>
			/// Retrieves MAPDX for the current row.
			/// </summary>
			public double _MAPDX
			{
				get
				{
					return System.Convert.ToDouble(m_DR[8]);
				}
			}
			private static double [] a_MAPDY = new double[ArraySize];
			/// <summary>
			/// Retrieves MAPDY for the current row.
			/// </summary>
			public double _MAPDY
			{
				get
				{
					return System.Convert.ToDouble(m_DR[9]);
				}
			}
			private static string [] a_MARKSETS = new string[ArraySize];
			/// <summary>
			/// Retrieves MARKSETS for the current row.
			/// </summary>
			public string _MARKSETS
			{
				get
				{
					return System.Convert.ToString(m_DR[10]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is actually inserted later, in a batch insertion command. The Flush method must be called to ensure all rows are actually written.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">the value to be inserted for ID_EVENTBRICK.</param>
			/// <param name="i_ID_PLATE">the value to be inserted for ID_PLATE.</param>
			/// <param name="i_ID_PROCESSOPERATION">the value to be inserted for ID_PROCESSOPERATION.</param>
			/// <param name="i_Z">the value to be inserted for Z.</param>
			/// <param name="i_MAPXX">the value to be inserted for MAPXX.</param>
			/// <param name="i_MAPXY">the value to be inserted for MAPXY.</param>
			/// <param name="i_MAPYX">the value to be inserted for MAPYX.</param>
			/// <param name="i_MAPYY">the value to be inserted for MAPYY.</param>
			/// <param name="i_MAPDX">the value to be inserted for MAPDX.</param>
			/// <param name="i_MAPDY">the value to be inserted for MAPDY.</param>
			/// <param name="i_MARKSETS">the value to be inserted for MARKSETS.</param>
			static public void Insert(long i_ID_EVENTBRICK,long i_ID_PLATE,long i_ID_PROCESSOPERATION,double i_Z,double i_MAPXX,double i_MAPXY,double i_MAPYX,double i_MAPYY,double i_MAPDX,double i_MAPDY,string i_MARKSETS)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				a_ID_EVENTBRICK[index] = i_ID_EVENTBRICK;
				a_ID_PLATE[index] = i_ID_PLATE;
				a_ID_PROCESSOPERATION[index] = i_ID_PROCESSOPERATION;
				a_Z[index] = i_Z;
				a_MAPXX[index] = i_MAPXX;
				a_MAPXY[index] = i_MAPXY;
				a_MAPYX[index] = i_MAPYX;
				a_MAPYY[index] = i_MAPYY;
				a_MAPDX[index] = i_MAPDX;
				a_MAPDY[index] = i_MAPDY;
				a_MARKSETS[index] = i_MARKSETS;
				if (++index >= ArraySize) { cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			}
			/// <summary>
			/// Reads a set of rows from TB_PLATE_CALIBRATIONS and retrieves them into a new TB_PLATE_CALIBRATIONS object.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_PLATE">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_PROCESSOPERATION">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="order">the ordering scheme to be applied to returned rows. This applies to all columns in the primary key.</param>
			/// <returns>a new instance of the TB_PLATE_CALIBRATIONS class that can be used to read the retrieved data.</returns>
			static public TB_PLATE_CALIBRATIONS SelectPrimaryKey(object i_ID_EVENTBRICK,object i_ID_PLATE,object i_ID_PROCESSOPERATION, OrderBy order)
			{
				string wherestr = "";
				string wtempstr = "";
				if (i_ID_EVENTBRICK != null)
				{
					if (i_ID_EVENTBRICK == System.DBNull.Value) wtempstr = "ID_EVENTBRICK IS NULL";
					else wtempstr = "ID_EVENTBRICK = " + i_ID_EVENTBRICK.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_PLATE != null)
				{
					if (i_ID_PLATE == System.DBNull.Value) wtempstr = "ID_PLATE IS NULL";
					else wtempstr = "ID_PLATE = " + i_ID_PLATE.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_PROCESSOPERATION != null)
				{
					if (i_ID_PROCESSOPERATION == System.DBNull.Value) wtempstr = "ID_PROCESSOPERATION IS NULL";
					else wtempstr = "ID_PROCESSOPERATION = " + i_ID_PROCESSOPERATION.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (order == OrderBy.Ascending) return SelectWhere(wherestr, "ID_EVENTBRICK ASC,ID_PLATE ASC,ID_PROCESSOPERATION ASC");
				else if (order == OrderBy.Descending) return SelectWhere(wherestr, "ID_EVENTBRICK DESC,ID_PLATE DESC,ID_PROCESSOPERATION DESC");
				return SelectWhere(wherestr, null);
			}
			/// <summary>
			/// Reads a set of rows from TB_PLATE_CALIBRATIONS and retrieves them into a new TB_PLATE_CALIBRATIONS object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the TB_PLATE_CALIBRATIONS class that can be used to read the retrieved data.</returns>
			static public TB_PLATE_CALIBRATIONS SelectWhere(string wherestr, string orderstr)
			{
				TB_PLATE_CALIBRATIONS newobj = new TB_PLATE_CALIBRATIONS();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_EVENTBRICK,ID_PLATE,ID_PROCESSOPERATION,Z,MAPXX,MAPXY,MAPYX,MAPYY,MAPDX,MAPDY,MARKSETS FROM TB_PLATE_CALIBRATIONS" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO TB_PLATE_CALIBRATIONS (ID_EVENTBRICK,ID_PLATE,ID_PROCESSOPERATION,Z,MAPXX,MAPXY,MAPYX,MAPYY,MAPDX,MAPDY,MARKSETS) VALUES (:p_1,:p_2,:p_3,:p_4,:p_5,:p_6,:p_7,:p_8,:p_9,:p_10,:p_11)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_EVENTBRICK;
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_PLATE;
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_PROCESSOPERATION;
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_Z;
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_MAPXX;
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_MAPXY;
				newcmd.Parameters.Add("p_7", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_MAPYX;
				newcmd.Parameters.Add("p_8", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_MAPYY;
				newcmd.Parameters.Add("p_9", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_MAPDX;
				newcmd.Parameters.Add("p_10", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_MAPDY;
				newcmd.Parameters.Add("p_11", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input).Value = a_MARKSETS;
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the TB_PLATE_DAMAGENOTICES table in the DB.
		/// For data insertion, the Insert and Flush static methods are used. This class inserts data in batches.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class TB_PLATE_DAMAGENOTICES
		{
			internal TB_PLATE_DAMAGENOTICES() {}
			System.Data.DataRowCollection m_DRC;
			const int ArraySize = 100;
			static int index = 0;
			/// <summary>
			/// Since this class uses batch insertion, this method must be called to ensure that all rows are actually inserted into TB_PLATE_DAMAGENOTICES. Failure to do so will result in incomplete writes.
			/// </summary>
			static public void Flush() { if (cmd == null || index == 0) return; cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			private static long [] a_ID_EVENTBRICK = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_EVENTBRICK for the current row.
			/// </summary>
			public long _ID_EVENTBRICK
			{
				get
				{
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			private static long [] a_ID_PLATE = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_PLATE for the current row.
			/// </summary>
			public long _ID_PLATE
			{
				get
				{
					return System.Convert.ToInt64(m_DR[1]);
				}
			}
			private static long [] a_ID_PROCESSOPERATION = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_PROCESSOPERATION for the current row.
			/// </summary>
			public long _ID_PROCESSOPERATION
			{
				get
				{
					return System.Convert.ToInt64(m_DR[2]);
				}
			}
			private static System.DateTime [] a_DETECTIONTIME = new System.DateTime[ArraySize];
			/// <summary>
			/// Retrieves DETECTIONTIME for the current row.
			/// </summary>
			public System.DateTime _DETECTIONTIME
			{
				get
				{
					return System.Convert.ToDateTime(m_DR[3]);
				}
			}
			private static char [] a_DAMAGED = new char[ArraySize];
			/// <summary>
			/// Retrieves DAMAGED for the current row.
			/// </summary>
			public char _DAMAGED
			{
				get
				{
					return System.Convert.ToChar(m_DR[4]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is actually inserted later, in a batch insertion command. The Flush method must be called to ensure all rows are actually written.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">the value to be inserted for ID_EVENTBRICK.</param>
			/// <param name="i_ID_PLATE">the value to be inserted for ID_PLATE.</param>
			/// <param name="i_ID_PROCESSOPERATION">the value to be inserted for ID_PROCESSOPERATION.</param>
			/// <param name="i_DETECTIONTIME">the value to be inserted for DETECTIONTIME.</param>
			/// <param name="i_DAMAGED">the value to be inserted for DAMAGED.</param>
			static public void Insert(long i_ID_EVENTBRICK,long i_ID_PLATE,long i_ID_PROCESSOPERATION,System.DateTime i_DETECTIONTIME,char i_DAMAGED)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				a_ID_EVENTBRICK[index] = i_ID_EVENTBRICK;
				a_ID_PLATE[index] = i_ID_PLATE;
				a_ID_PROCESSOPERATION[index] = i_ID_PROCESSOPERATION;
				a_DETECTIONTIME[index] = i_DETECTIONTIME;
				a_DAMAGED[index] = i_DAMAGED;
				if (++index >= ArraySize) { cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			}
			/// <summary>
			/// Reads a set of rows from TB_PLATE_DAMAGENOTICES and retrieves them into a new TB_PLATE_DAMAGENOTICES object.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_PLATE">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_PROCESSOPERATION">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="order">the ordering scheme to be applied to returned rows. This applies to all columns in the primary key.</param>
			/// <returns>a new instance of the TB_PLATE_DAMAGENOTICES class that can be used to read the retrieved data.</returns>
			static public TB_PLATE_DAMAGENOTICES SelectPrimaryKey(object i_ID_EVENTBRICK,object i_ID_PLATE,object i_ID_PROCESSOPERATION, OrderBy order)
			{
				string wherestr = "";
				string wtempstr = "";
				if (i_ID_EVENTBRICK != null)
				{
					if (i_ID_EVENTBRICK == System.DBNull.Value) wtempstr = "ID_EVENTBRICK IS NULL";
					else wtempstr = "ID_EVENTBRICK = " + i_ID_EVENTBRICK.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_PLATE != null)
				{
					if (i_ID_PLATE == System.DBNull.Value) wtempstr = "ID_PLATE IS NULL";
					else wtempstr = "ID_PLATE = " + i_ID_PLATE.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_PROCESSOPERATION != null)
				{
					if (i_ID_PROCESSOPERATION == System.DBNull.Value) wtempstr = "ID_PROCESSOPERATION IS NULL";
					else wtempstr = "ID_PROCESSOPERATION = " + i_ID_PROCESSOPERATION.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (order == OrderBy.Ascending) return SelectWhere(wherestr, "ID_EVENTBRICK ASC,ID_PLATE ASC,ID_PROCESSOPERATION ASC");
				else if (order == OrderBy.Descending) return SelectWhere(wherestr, "ID_EVENTBRICK DESC,ID_PLATE DESC,ID_PROCESSOPERATION DESC");
				return SelectWhere(wherestr, null);
			}
			/// <summary>
			/// Reads a set of rows from TB_PLATE_DAMAGENOTICES and retrieves them into a new TB_PLATE_DAMAGENOTICES object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the TB_PLATE_DAMAGENOTICES class that can be used to read the retrieved data.</returns>
			static public TB_PLATE_DAMAGENOTICES SelectWhere(string wherestr, string orderstr)
			{
				TB_PLATE_DAMAGENOTICES newobj = new TB_PLATE_DAMAGENOTICES();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_EVENTBRICK,ID_PLATE,ID_PROCESSOPERATION,DETECTIONTIME,DAMAGED FROM TB_PLATE_DAMAGENOTICES" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO TB_PLATE_DAMAGENOTICES (ID_EVENTBRICK,ID_PLATE,ID_PROCESSOPERATION,DETECTIONTIME,DAMAGED) VALUES (:p_1,:p_2,:p_3,:p_4,:p_5)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_EVENTBRICK;
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_PLATE;
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_PROCESSOPERATION;
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.DateTime, System.Data.ParameterDirection.Input).Value = a_DETECTIONTIME;
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input).Value = a_DAMAGED;
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the TB_PREDICTED_BRICKS table in the DB.
		/// For data insertion, the Insert and Flush static methods are used. This class inserts data in batches.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class TB_PREDICTED_BRICKS
		{
			internal TB_PREDICTED_BRICKS() {}
			System.Data.DataRowCollection m_DRC;
			const int ArraySize = 100;
			static int index = 0;
			/// <summary>
			/// Since this class uses batch insertion, this method must be called to ensure that all rows are actually inserted into TB_PREDICTED_BRICKS. Failure to do so will result in incomplete writes.
			/// </summary>
			static public void Flush() { if (cmd == null || index == 0) return; cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			private static long [] a_ID_EVENT = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_EVENT for the current row.
			/// </summary>
			public long _ID_EVENT
			{
				get
				{
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			private static long [] a_ID_EVENTBRICK = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_EVENTBRICK for the current row.
			/// </summary>
			public long _ID_EVENTBRICK
			{
				get
				{
					return System.Convert.ToInt64(m_DR[1]);
				}
			}
			private static double [] a_PROBABILITY = new double[ArraySize];
			/// <summary>
			/// Retrieves PROBABILITY for the current row.
			/// </summary>
			public double _PROBABILITY
			{
				get
				{
					return System.Convert.ToDouble(m_DR[2]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is actually inserted later, in a batch insertion command. The Flush method must be called to ensure all rows are actually written.
			/// </summary>
			/// <param name="i_ID_EVENT">the value to be inserted for ID_EVENT.</param>
			/// <param name="i_ID_EVENTBRICK">the value to be inserted for ID_EVENTBRICK.</param>
			/// <param name="i_PROBABILITY">the value to be inserted for PROBABILITY.</param>
			static public void Insert(long i_ID_EVENT,long i_ID_EVENTBRICK,double i_PROBABILITY)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				a_ID_EVENT[index] = i_ID_EVENT;
				a_ID_EVENTBRICK[index] = i_ID_EVENTBRICK;
				a_PROBABILITY[index] = i_PROBABILITY;
				if (++index >= ArraySize) { cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			}
			/// <summary>
			/// Reads a set of rows from TB_PREDICTED_BRICKS and retrieves them into a new TB_PREDICTED_BRICKS object.
			/// </summary>
			/// <param name="i_ID_EVENT">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_EVENTBRICK">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="order">the ordering scheme to be applied to returned rows. This applies to all columns in the primary key.</param>
			/// <returns>a new instance of the TB_PREDICTED_BRICKS class that can be used to read the retrieved data.</returns>
			static public TB_PREDICTED_BRICKS SelectPrimaryKey(object i_ID_EVENT,object i_ID_EVENTBRICK, OrderBy order)
			{
				string wherestr = "";
				string wtempstr = "";
				if (i_ID_EVENT != null)
				{
					if (i_ID_EVENT == System.DBNull.Value) wtempstr = "ID_EVENT IS NULL";
					else wtempstr = "ID_EVENT = " + i_ID_EVENT.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_EVENTBRICK != null)
				{
					if (i_ID_EVENTBRICK == System.DBNull.Value) wtempstr = "ID_EVENTBRICK IS NULL";
					else wtempstr = "ID_EVENTBRICK = " + i_ID_EVENTBRICK.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (order == OrderBy.Ascending) return SelectWhere(wherestr, "ID_EVENT ASC,ID_EVENTBRICK ASC");
				else if (order == OrderBy.Descending) return SelectWhere(wherestr, "ID_EVENT DESC,ID_EVENTBRICK DESC");
				return SelectWhere(wherestr, null);
			}
			/// <summary>
			/// Reads a set of rows from TB_PREDICTED_BRICKS and retrieves them into a new TB_PREDICTED_BRICKS object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the TB_PREDICTED_BRICKS class that can be used to read the retrieved data.</returns>
			static public TB_PREDICTED_BRICKS SelectWhere(string wherestr, string orderstr)
			{
				TB_PREDICTED_BRICKS newobj = new TB_PREDICTED_BRICKS();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_EVENT,ID_EVENTBRICK,PROBABILITY FROM TB_PREDICTED_BRICKS" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO TB_PREDICTED_BRICKS (ID_EVENT,ID_EVENTBRICK,PROBABILITY) VALUES (:p_1,:p_2,:p_3)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_EVENT;
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_EVENTBRICK;
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_PROBABILITY;
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the TB_PREDICTED_EVENTS table in the DB.
		/// For data insertion, the Insert method is used. Rows are inserted one by one.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class TB_PREDICTED_EVENTS
		{
			internal TB_PREDICTED_EVENTS() {}
			System.Data.DataRowCollection m_DRC;
			/// <summary>
			/// Retrieves EVENT for the current row.
			/// </summary>
			public long _EVENT
			{
				get
				{
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			/// <summary>
			/// Retrieves ID_PROCESSOPERATION for the current row.
			/// </summary>
			public long _ID_PROCESSOPERATION
			{
				get
				{
					return System.Convert.ToInt64(m_DR[1]);
				}
			}
			/// <summary>
			/// Retrieves ID for the current row.
			/// </summary>
			public long _ID
			{
				get
				{
					return System.Convert.ToInt64(m_DR[2]);
				}
			}
			/// <summary>
			/// Retrieves POSX for the current row.
			/// </summary>
			public double _POSX
			{
				get
				{
					return System.Convert.ToDouble(m_DR[3]);
				}
			}
			/// <summary>
			/// Retrieves POSY for the current row.
			/// </summary>
			public double _POSY
			{
				get
				{
					return System.Convert.ToDouble(m_DR[4]);
				}
			}
			/// <summary>
			/// Retrieves POSZ for the current row.
			/// </summary>
			public double _POSZ
			{
				get
				{
					return System.Convert.ToDouble(m_DR[5]);
				}
			}
			/// <summary>
			/// Retrieves TYPE for the current row.
			/// </summary>
			public string _TYPE
			{
				get
				{
					return System.Convert.ToString(m_DR[6]);
				}
			}
			/// <summary>
			/// Retrieves TIME for the current row.
			/// </summary>
			public System.DateTime _TIME
			{
				get
				{
					return System.Convert.ToDateTime(m_DR[7]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is inserted immediately.
			/// </summary>
			/// <param name="i_EVENT">the value to be inserted for EVENT.</param>
			/// <param name="i_ID_PROCESSOPERATION">the value to be inserted for ID_PROCESSOPERATION.</param>
			/// <param name="i_ID">the value to be inserted for ID. This value is actually used only if this method call is involved in data publication/replication, otherwise the actual value is generated by the OPERA DB and the supplied value is ignored.</param>
			/// <param name="i_POSX">the value to be inserted for POSX.</param>
			/// <param name="i_POSY">the value to be inserted for POSY.</param>
			/// <param name="i_POSZ">the value to be inserted for POSZ.</param>
			/// <param name="i_TYPE">the value to be inserted for TYPE.</param>
			/// <param name="i_TIME">the value to be inserted for TIME.</param>
			/// <returns>the value of ID for the new row.</returns>
			static public long Insert(long i_EVENT,long i_ID_PROCESSOPERATION,long i_ID,double i_POSX,double i_POSY,double i_POSZ,string i_TYPE,System.DateTime i_TIME)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = i_EVENT;
				cmd.Parameters[1].Value = i_ID_PROCESSOPERATION;
				cmd.Parameters[2].Value = i_ID;
				cmd.Parameters[3].Value = i_POSX;
				cmd.Parameters[4].Value = i_POSY;
				cmd.Parameters[5].Value = i_POSZ;
				cmd.Parameters[6].Value = i_TYPE;
				cmd.Parameters[7].Value = i_TIME;
				cmd.ExecuteNonQuery();
				return SySal.OperaDb.Convert.ToInt64(cmd.Parameters[8].Value);
			}
			/// <summary>
			/// Reads a set of rows from TB_PREDICTED_EVENTS and retrieves them into a new TB_PREDICTED_EVENTS object.
			/// </summary>
			/// <param name="i_EVENT">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_PROCESSOPERATION">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="order">the ordering scheme to be applied to returned rows. This applies to all columns in the primary key.</param>
			/// <returns>a new instance of the TB_PREDICTED_EVENTS class that can be used to read the retrieved data.</returns>
			static public TB_PREDICTED_EVENTS SelectPrimaryKey(object i_EVENT,object i_ID_PROCESSOPERATION, OrderBy order)
			{
				string wherestr = "";
				string wtempstr = "";
				if (i_EVENT != null)
				{
					if (i_EVENT == System.DBNull.Value) wtempstr = "EVENT IS NULL";
					else wtempstr = "EVENT = " + i_EVENT.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_PROCESSOPERATION != null)
				{
					if (i_ID_PROCESSOPERATION == System.DBNull.Value) wtempstr = "ID_PROCESSOPERATION IS NULL";
					else wtempstr = "ID_PROCESSOPERATION = " + i_ID_PROCESSOPERATION.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (order == OrderBy.Ascending) return SelectWhere(wherestr, "EVENT ASC,ID_PROCESSOPERATION ASC");
				else if (order == OrderBy.Descending) return SelectWhere(wherestr, "EVENT DESC,ID_PROCESSOPERATION DESC");
				return SelectWhere(wherestr, null);
			}
			/// <summary>
			/// Reads a set of rows from TB_PREDICTED_EVENTS and retrieves them into a new TB_PREDICTED_EVENTS object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the TB_PREDICTED_EVENTS class that can be used to read the retrieved data.</returns>
			static public TB_PREDICTED_EVENTS SelectWhere(string wherestr, string orderstr)
			{
				TB_PREDICTED_EVENTS newobj = new TB_PREDICTED_EVENTS();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT EVENT,ID_PROCESSOPERATION,ID,POSX,POSY,POSZ,TYPE,TIME FROM TB_PREDICTED_EVENTS" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO TB_PREDICTED_EVENTS (EVENT,ID_PROCESSOPERATION,ID,POSX,POSY,POSZ,TYPE,TIME) VALUES (:p_1,:p_2,:p_3,:p_4,:p_5,:p_6,:p_7,:p_8) RETURNING ID INTO :o_8");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_7", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_8", SySal.OperaDb.OperaDbType.DateTime, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("o_8", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Output);
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the TB_PREDICTED_TRACKS table in the DB.
		/// For data insertion, the Insert and Flush static methods are used. This class inserts data in batches.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class TB_PREDICTED_TRACKS
		{
			internal TB_PREDICTED_TRACKS() {}
			System.Data.DataRowCollection m_DRC;
			const int ArraySize = 100;
			static int index = 0;
			/// <summary>
			/// Since this class uses batch insertion, this method must be called to ensure that all rows are actually inserted into TB_PREDICTED_TRACKS. Failure to do so will result in incomplete writes.
			/// </summary>
			static public void Flush() { if (cmd == null || index == 0) return; cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			private static long [] a_ID_EVENT = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_EVENT for the current row.
			/// </summary>
			public long _ID_EVENT
			{
				get
				{
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			private static long [] a_TRACK = new long[ArraySize];
			/// <summary>
			/// Retrieves TRACK for the current row.
			/// </summary>
			public long _TRACK
			{
				get
				{
					return System.Convert.ToInt64(m_DR[1]);
				}
			}
			private static double [] a_POSZ = new double[ArraySize];
			/// <summary>
			/// Retrieves POSZ for the current row.
			/// </summary>
			public double _POSZ
			{
				get
				{
					return System.Convert.ToDouble(m_DR[2]);
				}
			}
			private static double [] a_POSX = new double[ArraySize];
			/// <summary>
			/// Retrieves POSX for the current row.
			/// </summary>
			public double _POSX
			{
				get
				{
					return System.Convert.ToDouble(m_DR[3]);
				}
			}
			private static double [] a_POSY = new double[ArraySize];
			/// <summary>
			/// Retrieves POSY for the current row.
			/// </summary>
			public double _POSY
			{
				get
				{
					return System.Convert.ToDouble(m_DR[4]);
				}
			}
			private static double [] a_SLOPEX = new double[ArraySize];
			/// <summary>
			/// Retrieves SLOPEX for the current row.
			/// </summary>
			public double _SLOPEX
			{
				get
				{
					return System.Convert.ToDouble(m_DR[5]);
				}
			}
			private static double [] a_SLOPEY = new double[ArraySize];
			/// <summary>
			/// Retrieves SLOPEY for the current row.
			/// </summary>
			public double _SLOPEY
			{
				get
				{
					return System.Convert.ToDouble(m_DR[6]);
				}
			}
			private static object [] a_FRAME = new object[ArraySize];
			/// <summary>
			/// Retrieves FRAME for the current row. The return value can be System.DBNull.Value or a value that can be cast to char.
			/// </summary>
			public object _FRAME
			{
				get
				{
					if (m_DR[7] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToChar(m_DR[7]);
				}
			}
			private static object [] a_POSTOL1 = new object[ArraySize];
			/// <summary>
			/// Retrieves POSTOL1 for the current row. The return value can be System.DBNull.Value or a value that can be cast to double.
			/// </summary>
			public object _POSTOL1
			{
				get
				{
					if (m_DR[8] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDouble(m_DR[8]);
				}
			}
			private static object [] a_POSTOL2 = new object[ArraySize];
			/// <summary>
			/// Retrieves POSTOL2 for the current row. The return value can be System.DBNull.Value or a value that can be cast to double.
			/// </summary>
			public object _POSTOL2
			{
				get
				{
					if (m_DR[9] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDouble(m_DR[9]);
				}
			}
			private static object [] a_SLOPETOL1 = new object[ArraySize];
			/// <summary>
			/// Retrieves SLOPETOL1 for the current row. The return value can be System.DBNull.Value or a value that can be cast to double.
			/// </summary>
			public object _SLOPETOL1
			{
				get
				{
					if (m_DR[10] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDouble(m_DR[10]);
				}
			}
			private static object [] a_SLOPETOL2 = new object[ArraySize];
			/// <summary>
			/// Retrieves SLOPETOL2 for the current row. The return value can be System.DBNull.Value or a value that can be cast to double.
			/// </summary>
			public object _SLOPETOL2
			{
				get
				{
					if (m_DR[11] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDouble(m_DR[11]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is actually inserted later, in a batch insertion command. The Flush method must be called to ensure all rows are actually written.
			/// </summary>
			/// <param name="i_ID_EVENT">the value to be inserted for ID_EVENT.</param>
			/// <param name="i_TRACK">the value to be inserted for TRACK.</param>
			/// <param name="i_POSZ">the value to be inserted for POSZ.</param>
			/// <param name="i_POSX">the value to be inserted for POSX.</param>
			/// <param name="i_POSY">the value to be inserted for POSY.</param>
			/// <param name="i_SLOPEX">the value to be inserted for SLOPEX.</param>
			/// <param name="i_SLOPEY">the value to be inserted for SLOPEY.</param>
			/// <param name="i_FRAME">the value to be inserted for FRAME. The value for this parameter can be char or System.DBNull.Value.</param>
			/// <param name="i_POSTOL1">the value to be inserted for POSTOL1. The value for this parameter can be double or System.DBNull.Value.</param>
			/// <param name="i_POSTOL2">the value to be inserted for POSTOL2. The value for this parameter can be double or System.DBNull.Value.</param>
			/// <param name="i_SLOPETOL1">the value to be inserted for SLOPETOL1. The value for this parameter can be double or System.DBNull.Value.</param>
			/// <param name="i_SLOPETOL2">the value to be inserted for SLOPETOL2. The value for this parameter can be double or System.DBNull.Value.</param>
			static public void Insert(long i_ID_EVENT,long i_TRACK,double i_POSZ,double i_POSX,double i_POSY,double i_SLOPEX,double i_SLOPEY,object i_FRAME,object i_POSTOL1,object i_POSTOL2,object i_SLOPETOL1,object i_SLOPETOL2)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				a_ID_EVENT[index] = i_ID_EVENT;
				a_TRACK[index] = i_TRACK;
				a_POSZ[index] = i_POSZ;
				a_POSX[index] = i_POSX;
				a_POSY[index] = i_POSY;
				a_SLOPEX[index] = i_SLOPEX;
				a_SLOPEY[index] = i_SLOPEY;
				a_FRAME[index] = i_FRAME;
				a_POSTOL1[index] = i_POSTOL1;
				a_POSTOL2[index] = i_POSTOL2;
				a_SLOPETOL1[index] = i_SLOPETOL1;
				a_SLOPETOL2[index] = i_SLOPETOL2;
				if (++index >= ArraySize) { cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			}
			/// <summary>
			/// Reads a set of rows from TB_PREDICTED_TRACKS and retrieves them into a new TB_PREDICTED_TRACKS object.
			/// </summary>
			/// <param name="i_ID_EVENT">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_TRACK">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="order">the ordering scheme to be applied to returned rows. This applies to all columns in the primary key.</param>
			/// <returns>a new instance of the TB_PREDICTED_TRACKS class that can be used to read the retrieved data.</returns>
			static public TB_PREDICTED_TRACKS SelectPrimaryKey(object i_ID_EVENT,object i_TRACK, OrderBy order)
			{
				string wherestr = "";
				string wtempstr = "";
				if (i_ID_EVENT != null)
				{
					if (i_ID_EVENT == System.DBNull.Value) wtempstr = "ID_EVENT IS NULL";
					else wtempstr = "ID_EVENT = " + i_ID_EVENT.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_TRACK != null)
				{
					if (i_TRACK == System.DBNull.Value) wtempstr = "TRACK IS NULL";
					else wtempstr = "TRACK = " + i_TRACK.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (order == OrderBy.Ascending) return SelectWhere(wherestr, "ID_EVENT ASC,TRACK ASC");
				else if (order == OrderBy.Descending) return SelectWhere(wherestr, "ID_EVENT DESC,TRACK DESC");
				return SelectWhere(wherestr, null);
			}
			/// <summary>
			/// Reads a set of rows from TB_PREDICTED_TRACKS and retrieves them into a new TB_PREDICTED_TRACKS object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the TB_PREDICTED_TRACKS class that can be used to read the retrieved data.</returns>
			static public TB_PREDICTED_TRACKS SelectWhere(string wherestr, string orderstr)
			{
				TB_PREDICTED_TRACKS newobj = new TB_PREDICTED_TRACKS();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_EVENT,TRACK,POSZ,POSX,POSY,SLOPEX,SLOPEY,FRAME,POSTOL1,POSTOL2,SLOPETOL1,SLOPETOL2 FROM TB_PREDICTED_TRACKS" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO TB_PREDICTED_TRACKS (ID_EVENT,TRACK,POSZ,POSX,POSY,SLOPEX,SLOPEY,FRAME,POSTOL1,POSTOL2,SLOPETOL1,SLOPETOL2) VALUES (:p_1,:p_2,:p_3,:p_4,:p_5,:p_6,:p_7,:p_8,:p_9,:p_10,:p_11,:p_12)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_EVENT;
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_TRACK;
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_POSZ;
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_POSX;
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_POSY;
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_SLOPEX;
				newcmd.Parameters.Add("p_7", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_SLOPEY;
				newcmd.Parameters.Add("p_8", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input).Value = a_FRAME;
				newcmd.Parameters.Add("p_9", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_POSTOL1;
				newcmd.Parameters.Add("p_10", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_POSTOL2;
				newcmd.Parameters.Add("p_11", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_SLOPETOL1;
				newcmd.Parameters.Add("p_12", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_SLOPETOL2;
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the TB_PRIVILEGES table in the DB.
		/// For data insertion, the Insert and Flush static methods are used. This class inserts data in batches.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class TB_PRIVILEGES
		{
			internal TB_PRIVILEGES() {}
			System.Data.DataRowCollection m_DRC;
			const int ArraySize = 100;
			static int index = 0;
			/// <summary>
			/// Since this class uses batch insertion, this method must be called to ensure that all rows are actually inserted into TB_PRIVILEGES. Failure to do so will result in incomplete writes.
			/// </summary>
			static public void Flush() { if (cmd == null || index == 0) return; cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			private static long [] a_ID_USER = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_USER for the current row.
			/// </summary>
			public long _ID_USER
			{
				get
				{
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			private static long [] a_ID_SITE = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_SITE for the current row.
			/// </summary>
			public long _ID_SITE
			{
				get
				{
					return System.Convert.ToInt64(m_DR[1]);
				}
			}
			private static int [] a_REQUESTSCAN = new int[ArraySize];
			/// <summary>
			/// Retrieves REQUESTSCAN for the current row.
			/// </summary>
			public int _REQUESTSCAN
			{
				get
				{
					return System.Convert.ToInt32(m_DR[2]);
				}
			}
			private static int [] a_REQUESTWEBANALYSIS = new int[ArraySize];
			/// <summary>
			/// Retrieves REQUESTWEBANALYSIS for the current row.
			/// </summary>
			public int _REQUESTWEBANALYSIS
			{
				get
				{
					return System.Convert.ToInt32(m_DR[3]);
				}
			}
			private static int [] a_REQUESTDATAPROCESSING = new int[ArraySize];
			/// <summary>
			/// Retrieves REQUESTDATAPROCESSING for the current row.
			/// </summary>
			public int _REQUESTDATAPROCESSING
			{
				get
				{
					return System.Convert.ToInt32(m_DR[4]);
				}
			}
			private static int [] a_REQUESTDATADOWNLOAD = new int[ArraySize];
			/// <summary>
			/// Retrieves REQUESTDATADOWNLOAD for the current row.
			/// </summary>
			public int _REQUESTDATADOWNLOAD
			{
				get
				{
					return System.Convert.ToInt32(m_DR[5]);
				}
			}
			private static int [] a_REQUESTPROCESSSTARTUP = new int[ArraySize];
			/// <summary>
			/// Retrieves REQUESTPROCESSSTARTUP for the current row.
			/// </summary>
			public int _REQUESTPROCESSSTARTUP
			{
				get
				{
					return System.Convert.ToInt32(m_DR[6]);
				}
			}
			private static int [] a_ADMINISTER = new int[ArraySize];
			/// <summary>
			/// Retrieves ADMINISTER for the current row.
			/// </summary>
			public int _ADMINISTER
			{
				get
				{
					return System.Convert.ToInt32(m_DR[7]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is actually inserted later, in a batch insertion command. The Flush method must be called to ensure all rows are actually written.
			/// </summary>
			/// <param name="i_ID_USER">the value to be inserted for ID_USER.</param>
			/// <param name="i_ID_SITE">the value to be inserted for ID_SITE.</param>
			/// <param name="i_REQUESTSCAN">the value to be inserted for REQUESTSCAN.</param>
			/// <param name="i_REQUESTWEBANALYSIS">the value to be inserted for REQUESTWEBANALYSIS.</param>
			/// <param name="i_REQUESTDATAPROCESSING">the value to be inserted for REQUESTDATAPROCESSING.</param>
			/// <param name="i_REQUESTDATADOWNLOAD">the value to be inserted for REQUESTDATADOWNLOAD.</param>
			/// <param name="i_REQUESTPROCESSSTARTUP">the value to be inserted for REQUESTPROCESSSTARTUP.</param>
			/// <param name="i_ADMINISTER">the value to be inserted for ADMINISTER.</param>
			static public void Insert(long i_ID_USER,long i_ID_SITE,int i_REQUESTSCAN,int i_REQUESTWEBANALYSIS,int i_REQUESTDATAPROCESSING,int i_REQUESTDATADOWNLOAD,int i_REQUESTPROCESSSTARTUP,int i_ADMINISTER)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				a_ID_USER[index] = i_ID_USER;
				a_ID_SITE[index] = i_ID_SITE;
				a_REQUESTSCAN[index] = i_REQUESTSCAN;
				a_REQUESTWEBANALYSIS[index] = i_REQUESTWEBANALYSIS;
				a_REQUESTDATAPROCESSING[index] = i_REQUESTDATAPROCESSING;
				a_REQUESTDATADOWNLOAD[index] = i_REQUESTDATADOWNLOAD;
				a_REQUESTPROCESSSTARTUP[index] = i_REQUESTPROCESSSTARTUP;
				a_ADMINISTER[index] = i_ADMINISTER;
				if (++index >= ArraySize) { cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			}
			/// <summary>
			/// Reads a set of rows from TB_PRIVILEGES and retrieves them into a new TB_PRIVILEGES object.
			/// </summary>
			/// <param name="i_ID_USER">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_SITE">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="order">the ordering scheme to be applied to returned rows. This applies to all columns in the primary key.</param>
			/// <returns>a new instance of the TB_PRIVILEGES class that can be used to read the retrieved data.</returns>
			static public TB_PRIVILEGES SelectPrimaryKey(object i_ID_USER,object i_ID_SITE, OrderBy order)
			{
				string wherestr = "";
				string wtempstr = "";
				if (i_ID_USER != null)
				{
					if (i_ID_USER == System.DBNull.Value) wtempstr = "ID_USER IS NULL";
					else wtempstr = "ID_USER = " + i_ID_USER.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_SITE != null)
				{
					if (i_ID_SITE == System.DBNull.Value) wtempstr = "ID_SITE IS NULL";
					else wtempstr = "ID_SITE = " + i_ID_SITE.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (order == OrderBy.Ascending) return SelectWhere(wherestr, "ID_USER ASC,ID_SITE ASC");
				else if (order == OrderBy.Descending) return SelectWhere(wherestr, "ID_USER DESC,ID_SITE DESC");
				return SelectWhere(wherestr, null);
			}
			/// <summary>
			/// Reads a set of rows from TB_PRIVILEGES and retrieves them into a new TB_PRIVILEGES object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the TB_PRIVILEGES class that can be used to read the retrieved data.</returns>
			static public TB_PRIVILEGES SelectWhere(string wherestr, string orderstr)
			{
				TB_PRIVILEGES newobj = new TB_PRIVILEGES();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_USER,ID_SITE,REQUESTSCAN,REQUESTWEBANALYSIS,REQUESTDATAPROCESSING,REQUESTDATADOWNLOAD,REQUESTPROCESSSTARTUP,ADMINISTER FROM TB_PRIVILEGES" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO TB_PRIVILEGES (ID_USER,ID_SITE,REQUESTSCAN,REQUESTWEBANALYSIS,REQUESTDATAPROCESSING,REQUESTDATADOWNLOAD,REQUESTPROCESSSTARTUP,ADMINISTER) VALUES (:p_1,:p_2,:p_3,:p_4,:p_5,:p_6,:p_7,:p_8)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_USER;
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_SITE;
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Int, System.Data.ParameterDirection.Input).Value = a_REQUESTSCAN;
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Int, System.Data.ParameterDirection.Input).Value = a_REQUESTWEBANALYSIS;
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Int, System.Data.ParameterDirection.Input).Value = a_REQUESTDATAPROCESSING;
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.Int, System.Data.ParameterDirection.Input).Value = a_REQUESTDATADOWNLOAD;
				newcmd.Parameters.Add("p_7", SySal.OperaDb.OperaDbType.Int, System.Data.ParameterDirection.Input).Value = a_REQUESTPROCESSSTARTUP;
				newcmd.Parameters.Add("p_8", SySal.OperaDb.OperaDbType.Int, System.Data.ParameterDirection.Input).Value = a_ADMINISTER;
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the TB_PROC_OPERATIONS table in the DB.
		/// For data insertion, the Insert method is used. Rows are inserted one by one.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class TB_PROC_OPERATIONS
		{
			internal TB_PROC_OPERATIONS() {}
			System.Data.DataRowCollection m_DRC;
			/// <summary>
			/// Retrieves ID for the current row.
			/// </summary>
			public long _ID
			{
				get
				{
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			/// <summary>
			/// Retrieves ID_MACHINE for the current row.
			/// </summary>
			public long _ID_MACHINE
			{
				get
				{
					return System.Convert.ToInt64(m_DR[1]);
				}
			}
			/// <summary>
			/// Retrieves ID_PROGRAMSETTINGS for the current row.
			/// </summary>
			public long _ID_PROGRAMSETTINGS
			{
				get
				{
					return System.Convert.ToInt64(m_DR[2]);
				}
			}
			/// <summary>
			/// Retrieves ID_REQUESTER for the current row.
			/// </summary>
			public long _ID_REQUESTER
			{
				get
				{
					return System.Convert.ToInt64(m_DR[3]);
				}
			}
			/// <summary>
			/// Retrieves ID_PARENT_OPERATION for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _ID_PARENT_OPERATION
			{
				get
				{
					if (m_DR[4] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[4]);
				}
			}
			/// <summary>
			/// Retrieves ID_EVENTBRICK for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _ID_EVENTBRICK
			{
				get
				{
					if (m_DR[5] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[5]);
				}
			}
			/// <summary>
			/// Retrieves ID_PLATE for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _ID_PLATE
			{
				get
				{
					if (m_DR[6] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[6]);
				}
			}
			/// <summary>
			/// Retrieves DRIVERLEVEL for the current row.
			/// </summary>
			public int _DRIVERLEVEL
			{
				get
				{
					return System.Convert.ToInt32(m_DR[7]);
				}
			}
			/// <summary>
			/// Retrieves ID_CALIBRATION_OPERATION for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _ID_CALIBRATION_OPERATION
			{
				get
				{
					if (m_DR[8] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[8]);
				}
			}
			/// <summary>
			/// Retrieves STARTTIME for the current row.
			/// </summary>
			public System.DateTime _STARTTIME
			{
				get
				{
					return System.Convert.ToDateTime(m_DR[9]);
				}
			}
			/// <summary>
			/// Retrieves FINISHTIME for the current row. The return value can be System.DBNull.Value or a value that can be cast to System.DateTime.
			/// </summary>
			public object _FINISHTIME
			{
				get
				{
					if (m_DR[10] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDateTime(m_DR[10]);
				}
			}
			/// <summary>
			/// Retrieves SUCCESS for the current row.
			/// </summary>
			public char _SUCCESS
			{
				get
				{
					return System.Convert.ToChar(m_DR[11]);
				}
			}
			/// <summary>
			/// Retrieves NOTES for the current row. The return value can be System.DBNull.Value or a value that can be cast to string.
			/// </summary>
			public object _NOTES
			{
				get
				{
					if (m_DR[12] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToString(m_DR[12]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is inserted immediately.
			/// </summary>
			/// <param name="i_ID">the value to be inserted for ID. This value is actually used only if this method call is involved in data publication/replication, otherwise the actual value is generated by the OPERA DB and the supplied value is ignored.</param>
			/// <param name="i_ID_MACHINE">the value to be inserted for ID_MACHINE.</param>
			/// <param name="i_ID_PROGRAMSETTINGS">the value to be inserted for ID_PROGRAMSETTINGS.</param>
			/// <param name="i_ID_REQUESTER">the value to be inserted for ID_REQUESTER.</param>
			/// <param name="i_ID_PARENT_OPERATION">the value to be inserted for ID_PARENT_OPERATION. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_ID_EVENTBRICK">the value to be inserted for ID_EVENTBRICK. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_ID_PLATE">the value to be inserted for ID_PLATE. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_DRIVERLEVEL">the value to be inserted for DRIVERLEVEL.</param>
			/// <param name="i_ID_CALIBRATION_OPERATION">the value to be inserted for ID_CALIBRATION_OPERATION. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_STARTTIME">the value to be inserted for STARTTIME.</param>
			/// <param name="i_FINISHTIME">the value to be inserted for FINISHTIME. The value for this parameter can be System.DateTime or System.DBNull.Value.</param>
			/// <param name="i_SUCCESS">the value to be inserted for SUCCESS.</param>
			/// <param name="i_NOTES">the value to be inserted for NOTES. The value for this parameter can be string or System.DBNull.Value.</param>
			/// <returns>the value of ID for the new row.</returns>
			static public long Insert(long i_ID,long i_ID_MACHINE,long i_ID_PROGRAMSETTINGS,long i_ID_REQUESTER,object i_ID_PARENT_OPERATION,object i_ID_EVENTBRICK,object i_ID_PLATE,int i_DRIVERLEVEL,object i_ID_CALIBRATION_OPERATION,System.DateTime i_STARTTIME,object i_FINISHTIME,char i_SUCCESS,object i_NOTES)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = i_ID;
				cmd.Parameters[1].Value = i_ID_MACHINE;
				cmd.Parameters[2].Value = i_ID_PROGRAMSETTINGS;
				cmd.Parameters[3].Value = i_ID_REQUESTER;
				cmd.Parameters[4].Value = i_ID_PARENT_OPERATION;
				cmd.Parameters[5].Value = i_ID_EVENTBRICK;
				cmd.Parameters[6].Value = i_ID_PLATE;
				cmd.Parameters[7].Value = i_DRIVERLEVEL;
				cmd.Parameters[8].Value = i_ID_CALIBRATION_OPERATION;
				cmd.Parameters[9].Value = i_STARTTIME;
				cmd.Parameters[10].Value = i_FINISHTIME;
				cmd.Parameters[11].Value = i_SUCCESS;
				cmd.Parameters[12].Value = i_NOTES;
				cmd.ExecuteNonQuery();
				return SySal.OperaDb.Convert.ToInt64(cmd.Parameters[13].Value);
			}
			/// <summary>
			/// Reads a set of rows from TB_PROC_OPERATIONS and retrieves them into a new TB_PROC_OPERATIONS object.
			/// </summary>
			/// <param name="i_ID">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="order">the ordering scheme to be applied to returned rows.</param>
			/// <returns>a new instance of the TB_PROC_OPERATIONS class that can be used to read the retrieved data.</returns>
			static public TB_PROC_OPERATIONS SelectPrimaryKey(object i_ID, OrderBy order)
			{
				string wherestr = "";
				string wtempstr = "";
				if (i_ID != null)
				{
					if (i_ID == System.DBNull.Value) wtempstr = "ID IS NULL";
					else wtempstr = "ID = " + i_ID.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (order == OrderBy.Ascending) return SelectWhere(wherestr, "ID ASC");
				else if (order == OrderBy.Descending) return SelectWhere(wherestr, "ID DESC");
				return SelectWhere(wherestr, null);
			}
			/// <summary>
			/// Reads a set of rows from TB_PROC_OPERATIONS and retrieves them into a new TB_PROC_OPERATIONS object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the TB_PROC_OPERATIONS class that can be used to read the retrieved data.</returns>
			static public TB_PROC_OPERATIONS SelectWhere(string wherestr, string orderstr)
			{
				TB_PROC_OPERATIONS newobj = new TB_PROC_OPERATIONS();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID,ID_MACHINE,ID_PROGRAMSETTINGS,ID_REQUESTER,ID_PARENT_OPERATION,ID_EVENTBRICK,ID_PLATE,DRIVERLEVEL,ID_CALIBRATION_OPERATION,STARTTIME,FINISHTIME,SUCCESS,NOTES FROM TB_PROC_OPERATIONS" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO TB_PROC_OPERATIONS (ID,ID_MACHINE,ID_PROGRAMSETTINGS,ID_REQUESTER,ID_PARENT_OPERATION,ID_EVENTBRICK,ID_PLATE,DRIVERLEVEL,ID_CALIBRATION_OPERATION,STARTTIME,FINISHTIME,SUCCESS,NOTES) VALUES (:p_1,:p_2,:p_3,:p_4,:p_5,:p_6,:p_7,:p_8,:p_9,:p_10,:p_11,:p_12,:p_13) RETURNING ID INTO :o_13");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_7", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_8", SySal.OperaDb.OperaDbType.Int, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_9", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_10", SySal.OperaDb.OperaDbType.DateTime, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_11", SySal.OperaDb.OperaDbType.DateTime, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_12", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_13", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("o_13", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Output);
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the TB_PROGRAMSETTINGS table in the DB.
		/// For data insertion, the Insert method is used. Rows are inserted one by one.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class TB_PROGRAMSETTINGS
		{
			internal TB_PROGRAMSETTINGS() {}
			System.Data.DataRowCollection m_DRC;
			/// <summary>
			/// Retrieves ID for the current row.
			/// </summary>
			public long _ID
			{
				get
				{
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			/// <summary>
			/// Retrieves DESCRIPTION for the current row.
			/// </summary>
			public string _DESCRIPTION
			{
				get
				{
					return System.Convert.ToString(m_DR[1]);
				}
			}
			/// <summary>
			/// Retrieves EXECUTABLE for the current row.
			/// </summary>
			public string _EXECUTABLE
			{
				get
				{
					return System.Convert.ToString(m_DR[2]);
				}
			}
			/// <summary>
			/// Retrieves SETTINGS for the current row.
			/// </summary>
			public string _SETTINGS
			{
				get
				{
					return System.Convert.ToString(m_DR[3]);
				}
			}
			/// <summary>
			/// Retrieves ID_AUTHOR for the current row.
			/// </summary>
			public long _ID_AUTHOR
			{
				get
				{
					return System.Convert.ToInt64(m_DR[4]);
				}
			}
			/// <summary>
			/// Retrieves DRIVERLEVEL for the current row.
			/// </summary>
			public int _DRIVERLEVEL
			{
				get
				{
					return System.Convert.ToInt32(m_DR[5]);
				}
			}
			/// <summary>
			/// Retrieves TEMPLATEMARKS for the current row. The return value can be System.DBNull.Value or a value that can be cast to int.
			/// </summary>
			public object _TEMPLATEMARKS
			{
				get
				{
					if (m_DR[6] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt32(m_DR[6]);
				}
			}
			/// <summary>
			/// Retrieves MARKSET for the current row. The return value can be System.DBNull.Value or a value that can be cast to string.
			/// </summary>
			public object _MARKSET
			{
				get
				{
					if (m_DR[7] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToString(m_DR[7]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is inserted immediately.
			/// </summary>
			/// <param name="i_ID">the value to be inserted for ID. This value is actually used only if this method call is involved in data publication/replication, otherwise the actual value is generated by the OPERA DB and the supplied value is ignored.</param>
			/// <param name="i_DESCRIPTION">the value to be inserted for DESCRIPTION.</param>
			/// <param name="i_EXECUTABLE">the value to be inserted for EXECUTABLE.</param>
			/// <param name="i_SETTINGS">the value to be inserted for SETTINGS.</param>
			/// <param name="i_ID_AUTHOR">the value to be inserted for ID_AUTHOR.</param>
			/// <param name="i_DRIVERLEVEL">the value to be inserted for DRIVERLEVEL.</param>
			/// <param name="i_TEMPLATEMARKS">the value to be inserted for TEMPLATEMARKS. The value for this parameter can be int or System.DBNull.Value.</param>
			/// <param name="i_MARKSET">the value to be inserted for MARKSET. The value for this parameter can be string or System.DBNull.Value.</param>
			/// <returns>the value of ID for the new row.</returns>
			static public long Insert(long i_ID,string i_DESCRIPTION,string i_EXECUTABLE,string i_SETTINGS,long i_ID_AUTHOR,int i_DRIVERLEVEL,object i_TEMPLATEMARKS,object i_MARKSET)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = i_ID;
				cmd.Parameters[1].Value = i_DESCRIPTION;
				cmd.Parameters[2].Value = i_EXECUTABLE;
				cmd.Parameters[3].Value = i_SETTINGS;
				cmd.Parameters[4].Value = i_ID_AUTHOR;
				cmd.Parameters[5].Value = i_DRIVERLEVEL;
				cmd.Parameters[6].Value = i_TEMPLATEMARKS;
				cmd.Parameters[7].Value = i_MARKSET;
				cmd.ExecuteNonQuery();
				return SySal.OperaDb.Convert.ToInt64(cmd.Parameters[8].Value);
			}
			/// <summary>
			/// Reads a set of rows from TB_PROGRAMSETTINGS and retrieves them into a new TB_PROGRAMSETTINGS object.
			/// </summary>
			/// <param name="i_DESCRIPTION">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_EXECUTABLE">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="order">the ordering scheme to be applied to returned rows. This applies to all columns in the primary key.</param>
			/// <returns>a new instance of the TB_PROGRAMSETTINGS class that can be used to read the retrieved data.</returns>
			static public TB_PROGRAMSETTINGS SelectPrimaryKey(object i_DESCRIPTION,object i_EXECUTABLE, OrderBy order)
			{
				string wherestr = "";
				string wtempstr = "";
				if (i_DESCRIPTION != null)
				{
					if (i_DESCRIPTION == System.DBNull.Value) wtempstr = "DESCRIPTION IS NULL";
					else wtempstr = "DESCRIPTION = " + i_DESCRIPTION.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_EXECUTABLE != null)
				{
					if (i_EXECUTABLE == System.DBNull.Value) wtempstr = "EXECUTABLE IS NULL";
					else wtempstr = "EXECUTABLE = " + i_EXECUTABLE.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (order == OrderBy.Ascending) return SelectWhere(wherestr, "DESCRIPTION ASC,EXECUTABLE ASC");
				else if (order == OrderBy.Descending) return SelectWhere(wherestr, "DESCRIPTION DESC,EXECUTABLE DESC");
				return SelectWhere(wherestr, null);
			}
			/// <summary>
			/// Reads a set of rows from TB_PROGRAMSETTINGS and retrieves them into a new TB_PROGRAMSETTINGS object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the TB_PROGRAMSETTINGS class that can be used to read the retrieved data.</returns>
			static public TB_PROGRAMSETTINGS SelectWhere(string wherestr, string orderstr)
			{
				TB_PROGRAMSETTINGS newobj = new TB_PROGRAMSETTINGS();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID,DESCRIPTION,EXECUTABLE,SETTINGS,ID_AUTHOR,DRIVERLEVEL,TEMPLATEMARKS,MARKSET FROM TB_PROGRAMSETTINGS" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO TB_PROGRAMSETTINGS (ID,DESCRIPTION,EXECUTABLE,SETTINGS,ID_AUTHOR,DRIVERLEVEL,TEMPLATEMARKS,MARKSET) VALUES (:p_1,:p_2,:p_3,:p_4,:p_5,:p_6,:p_7,:p_8) RETURNING ID INTO :o_8");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.CLOB, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.Int, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_7", SySal.OperaDb.OperaDbType.Int, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_8", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("o_8", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Output);
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the TB_RECONSTRUCTIONS table in the DB.
		/// For data insertion, the Insert method is used. Rows are inserted one by one.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class TB_RECONSTRUCTIONS
		{
			internal TB_RECONSTRUCTIONS() {}
			System.Data.DataRowCollection m_DRC;
			/// <summary>
			/// Retrieves ID_EVENTBRICK for the current row.
			/// </summary>
			public long _ID_EVENTBRICK
			{
				get
				{
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			/// <summary>
			/// Retrieves ID for the current row.
			/// </summary>
			public long _ID
			{
				get
				{
					return System.Convert.ToInt64(m_DR[1]);
				}
			}
			/// <summary>
			/// Retrieves ID_PROCESSOPERATION for the current row.
			/// </summary>
			public long _ID_PROCESSOPERATION
			{
				get
				{
					return System.Convert.ToInt64(m_DR[2]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is inserted immediately.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">the value to be inserted for ID_EVENTBRICK.</param>
			/// <param name="i_ID">the value to be inserted for ID. This value is actually used only if this method call is involved in data publication/replication, otherwise the actual value is generated by the OPERA DB and the supplied value is ignored.</param>
			/// <param name="i_ID_PROCESSOPERATION">the value to be inserted for ID_PROCESSOPERATION.</param>
			/// <returns>the value of ID for the new row.</returns>
			static public long Insert(long i_ID_EVENTBRICK,long i_ID,long i_ID_PROCESSOPERATION)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = i_ID_EVENTBRICK;
				cmd.Parameters[1].Value = i_ID;
				cmd.Parameters[2].Value = i_ID_PROCESSOPERATION;
				cmd.ExecuteNonQuery();
				return SySal.OperaDb.Convert.ToInt64(cmd.Parameters[3].Value);
			}
			/// <summary>
			/// Reads a set of rows from TB_RECONSTRUCTIONS and retrieves them into a new TB_RECONSTRUCTIONS object.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="order">the ordering scheme to be applied to returned rows. This applies to all columns in the primary key.</param>
			/// <returns>a new instance of the TB_RECONSTRUCTIONS class that can be used to read the retrieved data.</returns>
			static public TB_RECONSTRUCTIONS SelectPrimaryKey(object i_ID_EVENTBRICK,object i_ID, OrderBy order)
			{
				string wherestr = "";
				string wtempstr = "";
				if (i_ID_EVENTBRICK != null)
				{
					if (i_ID_EVENTBRICK == System.DBNull.Value) wtempstr = "ID_EVENTBRICK IS NULL";
					else wtempstr = "ID_EVENTBRICK = " + i_ID_EVENTBRICK.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID != null)
				{
					if (i_ID == System.DBNull.Value) wtempstr = "ID IS NULL";
					else wtempstr = "ID = " + i_ID.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (order == OrderBy.Ascending) return SelectWhere(wherestr, "ID_EVENTBRICK ASC,ID ASC");
				else if (order == OrderBy.Descending) return SelectWhere(wherestr, "ID_EVENTBRICK DESC,ID DESC");
				return SelectWhere(wherestr, null);
			}
			/// <summary>
			/// Reads a set of rows from TB_RECONSTRUCTIONS and retrieves them into a new TB_RECONSTRUCTIONS object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the TB_RECONSTRUCTIONS class that can be used to read the retrieved data.</returns>
			static public TB_RECONSTRUCTIONS SelectWhere(string wherestr, string orderstr)
			{
				TB_RECONSTRUCTIONS newobj = new TB_RECONSTRUCTIONS();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_EVENTBRICK,ID,ID_PROCESSOPERATION FROM TB_RECONSTRUCTIONS" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO TB_RECONSTRUCTIONS (ID_EVENTBRICK,ID,ID_PROCESSOPERATION) VALUES (:p_1,:p_2,:p_3) RETURNING ID INTO :o_3");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("o_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Output);
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the TB_RECONSTRUCTION_LISTS table in the DB.
		/// For data insertion, the Insert and Flush static methods are used. This class inserts data in batches.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class TB_RECONSTRUCTION_LISTS
		{
			internal TB_RECONSTRUCTION_LISTS() {}
			System.Data.DataRowCollection m_DRC;
			const int ArraySize = 100;
			static int index = 0;
			/// <summary>
			/// Since this class uses batch insertion, this method must be called to ensure that all rows are actually inserted into TB_RECONSTRUCTION_LISTS. Failure to do so will result in incomplete writes.
			/// </summary>
			static public void Flush() { if (cmd == null || index == 0) return; cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			private static long [] a_ID_SET = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_SET for the current row.
			/// </summary>
			public long _ID_SET
			{
				get
				{
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			private static long [] a_ID_EVENTBRICK = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_EVENTBRICK for the current row.
			/// </summary>
			public long _ID_EVENTBRICK
			{
				get
				{
					return System.Convert.ToInt64(m_DR[1]);
				}
			}
			private static long [] a_ID_RECONSTRUCTION = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_RECONSTRUCTION for the current row.
			/// </summary>
			public long _ID_RECONSTRUCTION
			{
				get
				{
					return System.Convert.ToInt64(m_DR[2]);
				}
			}
			private static long [] a_ID_OPTION = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_OPTION for the current row.
			/// </summary>
			public long _ID_OPTION
			{
				get
				{
					return System.Convert.ToInt64(m_DR[3]);
				}
			}
			private static object [] a_NOTES = new object[ArraySize];
			/// <summary>
			/// Retrieves NOTES for the current row. The return value can be System.DBNull.Value or a value that can be cast to string.
			/// </summary>
			public object _NOTES
			{
				get
				{
					if (m_DR[4] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToString(m_DR[4]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is actually inserted later, in a batch insertion command. The Flush method must be called to ensure all rows are actually written.
			/// </summary>
			/// <param name="i_ID_SET">the value to be inserted for ID_SET.</param>
			/// <param name="i_ID_EVENTBRICK">the value to be inserted for ID_EVENTBRICK.</param>
			/// <param name="i_ID_RECONSTRUCTION">the value to be inserted for ID_RECONSTRUCTION.</param>
			/// <param name="i_ID_OPTION">the value to be inserted for ID_OPTION.</param>
			/// <param name="i_NOTES">the value to be inserted for NOTES. The value for this parameter can be string or System.DBNull.Value.</param>
			static public void Insert(long i_ID_SET,long i_ID_EVENTBRICK,long i_ID_RECONSTRUCTION,long i_ID_OPTION,object i_NOTES)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				a_ID_SET[index] = i_ID_SET;
				a_ID_EVENTBRICK[index] = i_ID_EVENTBRICK;
				a_ID_RECONSTRUCTION[index] = i_ID_RECONSTRUCTION;
				a_ID_OPTION[index] = i_ID_OPTION;
				a_NOTES[index] = i_NOTES;
				if (++index >= ArraySize) { cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			}
			/// <summary>
			/// Reads a set of rows from TB_RECONSTRUCTION_LISTS and retrieves them into a new TB_RECONSTRUCTION_LISTS object.
			/// </summary>
			/// <param name="i_ID_SET">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_EVENTBRICK">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_RECONSTRUCTION">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_OPTION">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="order">the ordering scheme to be applied to returned rows. This applies to all columns in the primary key.</param>
			/// <returns>a new instance of the TB_RECONSTRUCTION_LISTS class that can be used to read the retrieved data.</returns>
			static public TB_RECONSTRUCTION_LISTS SelectPrimaryKey(object i_ID_SET,object i_ID_EVENTBRICK,object i_ID_RECONSTRUCTION,object i_ID_OPTION, OrderBy order)
			{
				string wherestr = "";
				string wtempstr = "";
				if (i_ID_SET != null)
				{
					if (i_ID_SET == System.DBNull.Value) wtempstr = "ID_SET IS NULL";
					else wtempstr = "ID_SET = " + i_ID_SET.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_EVENTBRICK != null)
				{
					if (i_ID_EVENTBRICK == System.DBNull.Value) wtempstr = "ID_EVENTBRICK IS NULL";
					else wtempstr = "ID_EVENTBRICK = " + i_ID_EVENTBRICK.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_RECONSTRUCTION != null)
				{
					if (i_ID_RECONSTRUCTION == System.DBNull.Value) wtempstr = "ID_RECONSTRUCTION IS NULL";
					else wtempstr = "ID_RECONSTRUCTION = " + i_ID_RECONSTRUCTION.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_OPTION != null)
				{
					if (i_ID_OPTION == System.DBNull.Value) wtempstr = "ID_OPTION IS NULL";
					else wtempstr = "ID_OPTION = " + i_ID_OPTION.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (order == OrderBy.Ascending) return SelectWhere(wherestr, "ID_SET ASC,ID_EVENTBRICK ASC,ID_RECONSTRUCTION ASC,ID_OPTION ASC");
				else if (order == OrderBy.Descending) return SelectWhere(wherestr, "ID_SET DESC,ID_EVENTBRICK DESC,ID_RECONSTRUCTION DESC,ID_OPTION DESC");
				return SelectWhere(wherestr, null);
			}
			/// <summary>
			/// Reads a set of rows from TB_RECONSTRUCTION_LISTS and retrieves them into a new TB_RECONSTRUCTION_LISTS object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the TB_RECONSTRUCTION_LISTS class that can be used to read the retrieved data.</returns>
			static public TB_RECONSTRUCTION_LISTS SelectWhere(string wherestr, string orderstr)
			{
				TB_RECONSTRUCTION_LISTS newobj = new TB_RECONSTRUCTION_LISTS();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_SET,ID_EVENTBRICK,ID_RECONSTRUCTION,ID_OPTION,NOTES FROM TB_RECONSTRUCTION_LISTS" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO TB_RECONSTRUCTION_LISTS (ID_SET,ID_EVENTBRICK,ID_RECONSTRUCTION,ID_OPTION,NOTES) VALUES (:p_1,:p_2,:p_3,:p_4,:p_5)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_SET;
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_EVENTBRICK;
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_RECONSTRUCTION;
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_OPTION;
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.CLOB, System.Data.ParameterDirection.Input).Value = a_NOTES;
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the TB_RECONSTRUCTION_OPTIONS table in the DB.
		/// For data insertion, the Insert method is used. Rows are inserted one by one.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class TB_RECONSTRUCTION_OPTIONS
		{
			internal TB_RECONSTRUCTION_OPTIONS() {}
			System.Data.DataRowCollection m_DRC;
			/// <summary>
			/// Retrieves ID_EVENTBRICK for the current row.
			/// </summary>
			public long _ID_EVENTBRICK
			{
				get
				{
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			/// <summary>
			/// Retrieves ID_RECONSTRUCTION for the current row.
			/// </summary>
			public long _ID_RECONSTRUCTION
			{
				get
				{
					return System.Convert.ToInt64(m_DR[1]);
				}
			}
			/// <summary>
			/// Retrieves ID for the current row.
			/// </summary>
			public long _ID
			{
				get
				{
					return System.Convert.ToInt64(m_DR[2]);
				}
			}
			/// <summary>
			/// Retrieves GEOMETRICAL_PROBABILITY for the current row. The return value can be System.DBNull.Value or a value that can be cast to double.
			/// </summary>
			public object _GEOMETRICAL_PROBABILITY
			{
				get
				{
					if (m_DR[3] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDouble(m_DR[3]);
				}
			}
			/// <summary>
			/// Retrieves PHYSICAL_PROBABILITY for the current row. The return value can be System.DBNull.Value or a value that can be cast to double.
			/// </summary>
			public object _PHYSICAL_PROBABILITY
			{
				get
				{
					if (m_DR[4] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDouble(m_DR[4]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is inserted immediately.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">the value to be inserted for ID_EVENTBRICK.</param>
			/// <param name="i_ID_RECONSTRUCTION">the value to be inserted for ID_RECONSTRUCTION.</param>
			/// <param name="i_ID">the value to be inserted for ID. This value is actually used only if this method call is involved in data publication/replication, otherwise the actual value is generated by the OPERA DB and the supplied value is ignored.</param>
			/// <param name="i_GEOMETRICAL_PROBABILITY">the value to be inserted for GEOMETRICAL_PROBABILITY. The value for this parameter can be double or System.DBNull.Value.</param>
			/// <param name="i_PHYSICAL_PROBABILITY">the value to be inserted for PHYSICAL_PROBABILITY. The value for this parameter can be double or System.DBNull.Value.</param>
			/// <returns>the value of ID for the new row.</returns>
			static public long Insert(long i_ID_EVENTBRICK,long i_ID_RECONSTRUCTION,long i_ID,object i_GEOMETRICAL_PROBABILITY,object i_PHYSICAL_PROBABILITY)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = i_ID_EVENTBRICK;
				cmd.Parameters[1].Value = i_ID_RECONSTRUCTION;
				cmd.Parameters[2].Value = i_ID;
				cmd.Parameters[3].Value = i_GEOMETRICAL_PROBABILITY;
				cmd.Parameters[4].Value = i_PHYSICAL_PROBABILITY;
				cmd.ExecuteNonQuery();
				return SySal.OperaDb.Convert.ToInt64(cmd.Parameters[5].Value);
			}
			/// <summary>
			/// Reads a set of rows from TB_RECONSTRUCTION_OPTIONS and retrieves them into a new TB_RECONSTRUCTION_OPTIONS object.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_RECONSTRUCTION">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="order">the ordering scheme to be applied to returned rows. This applies to all columns in the primary key.</param>
			/// <returns>a new instance of the TB_RECONSTRUCTION_OPTIONS class that can be used to read the retrieved data.</returns>
			static public TB_RECONSTRUCTION_OPTIONS SelectPrimaryKey(object i_ID_EVENTBRICK,object i_ID_RECONSTRUCTION,object i_ID, OrderBy order)
			{
				string wherestr = "";
				string wtempstr = "";
				if (i_ID_EVENTBRICK != null)
				{
					if (i_ID_EVENTBRICK == System.DBNull.Value) wtempstr = "ID_EVENTBRICK IS NULL";
					else wtempstr = "ID_EVENTBRICK = " + i_ID_EVENTBRICK.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_RECONSTRUCTION != null)
				{
					if (i_ID_RECONSTRUCTION == System.DBNull.Value) wtempstr = "ID_RECONSTRUCTION IS NULL";
					else wtempstr = "ID_RECONSTRUCTION = " + i_ID_RECONSTRUCTION.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID != null)
				{
					if (i_ID == System.DBNull.Value) wtempstr = "ID IS NULL";
					else wtempstr = "ID = " + i_ID.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (order == OrderBy.Ascending) return SelectWhere(wherestr, "ID_EVENTBRICK ASC,ID_RECONSTRUCTION ASC,ID ASC");
				else if (order == OrderBy.Descending) return SelectWhere(wherestr, "ID_EVENTBRICK DESC,ID_RECONSTRUCTION DESC,ID DESC");
				return SelectWhere(wherestr, null);
			}
			/// <summary>
			/// Reads a set of rows from TB_RECONSTRUCTION_OPTIONS and retrieves them into a new TB_RECONSTRUCTION_OPTIONS object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the TB_RECONSTRUCTION_OPTIONS class that can be used to read the retrieved data.</returns>
			static public TB_RECONSTRUCTION_OPTIONS SelectWhere(string wherestr, string orderstr)
			{
				TB_RECONSTRUCTION_OPTIONS newobj = new TB_RECONSTRUCTION_OPTIONS();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_EVENTBRICK,ID_RECONSTRUCTION,ID,GEOMETRICAL_PROBABILITY,PHYSICAL_PROBABILITY FROM TB_RECONSTRUCTION_OPTIONS" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO TB_RECONSTRUCTION_OPTIONS (ID_EVENTBRICK,ID_RECONSTRUCTION,ID,GEOMETRICAL_PROBABILITY,PHYSICAL_PROBABILITY) VALUES (:p_1,:p_2,:p_3,:p_4,:p_5) RETURNING ID INTO :o_5");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("o_5", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Output);
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the TB_RECONSTRUCTION_SETS table in the DB.
		/// For data insertion, the Insert method is used. Rows are inserted one by one.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class TB_RECONSTRUCTION_SETS
		{
			internal TB_RECONSTRUCTION_SETS() {}
			System.Data.DataRowCollection m_DRC;
			/// <summary>
			/// Retrieves ID for the current row.
			/// </summary>
			public long _ID
			{
				get
				{
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			/// <summary>
			/// Retrieves RELEASETIME for the current row.
			/// </summary>
			public System.DateTime _RELEASETIME
			{
				get
				{
					return System.Convert.ToDateTime(m_DR[1]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is inserted immediately.
			/// </summary>
			/// <param name="i_ID">the value to be inserted for ID. This value is actually used only if this method call is involved in data publication/replication, otherwise the actual value is generated by the OPERA DB and the supplied value is ignored.</param>
			/// <param name="i_RELEASETIME">the value to be inserted for RELEASETIME.</param>
			/// <returns>the value of ID for the new row.</returns>
			static public long Insert(long i_ID,System.DateTime i_RELEASETIME)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = i_ID;
				cmd.Parameters[1].Value = i_RELEASETIME;
				cmd.ExecuteNonQuery();
				return SySal.OperaDb.Convert.ToInt64(cmd.Parameters[2].Value);
			}
			/// <summary>
			/// Reads a set of rows from TB_RECONSTRUCTION_SETS and retrieves them into a new TB_RECONSTRUCTION_SETS object.
			/// </summary>
			/// <param name="i_ID">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="order">the ordering scheme to be applied to returned rows.</param>
			/// <returns>a new instance of the TB_RECONSTRUCTION_SETS class that can be used to read the retrieved data.</returns>
			static public TB_RECONSTRUCTION_SETS SelectPrimaryKey(object i_ID, OrderBy order)
			{
				string wherestr = "";
				string wtempstr = "";
				if (i_ID != null)
				{
					if (i_ID == System.DBNull.Value) wtempstr = "ID IS NULL";
					else wtempstr = "ID = " + i_ID.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (order == OrderBy.Ascending) return SelectWhere(wherestr, "ID ASC");
				else if (order == OrderBy.Descending) return SelectWhere(wherestr, "ID DESC");
				return SelectWhere(wherestr, null);
			}
			/// <summary>
			/// Reads a set of rows from TB_RECONSTRUCTION_SETS and retrieves them into a new TB_RECONSTRUCTION_SETS object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the TB_RECONSTRUCTION_SETS class that can be used to read the retrieved data.</returns>
			static public TB_RECONSTRUCTION_SETS SelectWhere(string wherestr, string orderstr)
			{
				TB_RECONSTRUCTION_SETS newobj = new TB_RECONSTRUCTION_SETS();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID,RELEASETIME FROM TB_RECONSTRUCTION_SETS" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO TB_RECONSTRUCTION_SETS (ID,RELEASETIME) VALUES (:p_1,:p_2) RETURNING ID INTO :o_2");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.DateTime, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("o_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Output);
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the TB_SCANBACK_CHECKRESULTS table in the DB.
		/// For data insertion, the Insert and Flush static methods are used. This class inserts data in batches.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class TB_SCANBACK_CHECKRESULTS
		{
			internal TB_SCANBACK_CHECKRESULTS() {}
			System.Data.DataRowCollection m_DRC;
			const int ArraySize = 100;
			static int index = 0;
			/// <summary>
			/// Since this class uses batch insertion, this method must be called to ensure that all rows are actually inserted into TB_SCANBACK_CHECKRESULTS. Failure to do so will result in incomplete writes.
			/// </summary>
			static public void Flush() { if (cmd == null || index == 0) return; cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			private static long [] a_ID_EVENTBRICK = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_EVENTBRICK for the current row.
			/// </summary>
			public long _ID_EVENTBRICK
			{
				get
				{
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			private static long [] a_ID_PROCESSOPERATION = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_PROCESSOPERATION for the current row.
			/// </summary>
			public long _ID_PROCESSOPERATION
			{
				get
				{
					return System.Convert.ToInt64(m_DR[1]);
				}
			}
			private static long [] a_ID_PATH = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_PATH for the current row.
			/// </summary>
			public long _ID_PATH
			{
				get
				{
					return System.Convert.ToInt64(m_DR[2]);
				}
			}
			private static object [] a_ID_PLATE = new object[ArraySize];
			/// <summary>
			/// Retrieves ID_PLATE for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _ID_PLATE
			{
				get
				{
					if (m_DR[3] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[3]);
				}
			}
			private static object [] a_RESULT = new object[ArraySize];
			/// <summary>
			/// Retrieves RESULT for the current row. The return value can be System.DBNull.Value or a value that can be cast to string.
			/// </summary>
			public object _RESULT
			{
				get
				{
					if (m_DR[4] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToString(m_DR[4]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is actually inserted later, in a batch insertion command. The Flush method must be called to ensure all rows are actually written.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">the value to be inserted for ID_EVENTBRICK.</param>
			/// <param name="i_ID_PROCESSOPERATION">the value to be inserted for ID_PROCESSOPERATION.</param>
			/// <param name="i_ID_PATH">the value to be inserted for ID_PATH.</param>
			/// <param name="i_ID_PLATE">the value to be inserted for ID_PLATE. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_RESULT">the value to be inserted for RESULT. The value for this parameter can be string or System.DBNull.Value.</param>
			static public void Insert(long i_ID_EVENTBRICK,long i_ID_PROCESSOPERATION,long i_ID_PATH,object i_ID_PLATE,object i_RESULT)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				a_ID_EVENTBRICK[index] = i_ID_EVENTBRICK;
				a_ID_PROCESSOPERATION[index] = i_ID_PROCESSOPERATION;
				a_ID_PATH[index] = i_ID_PATH;
				a_ID_PLATE[index] = i_ID_PLATE;
				a_RESULT[index] = i_RESULT;
				if (++index >= ArraySize) { cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			}
			/// <summary>
			/// Reads a set of rows from TB_SCANBACK_CHECKRESULTS and retrieves them into a new TB_SCANBACK_CHECKRESULTS object.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_PROCESSOPERATION">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_PATH">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="order">the ordering scheme to be applied to returned rows. This applies to all columns in the primary key.</param>
			/// <returns>a new instance of the TB_SCANBACK_CHECKRESULTS class that can be used to read the retrieved data.</returns>
			static public TB_SCANBACK_CHECKRESULTS SelectPrimaryKey(object i_ID_EVENTBRICK,object i_ID_PROCESSOPERATION,object i_ID_PATH, OrderBy order)
			{
				string wherestr = "";
				string wtempstr = "";
				if (i_ID_EVENTBRICK != null)
				{
					if (i_ID_EVENTBRICK == System.DBNull.Value) wtempstr = "ID_EVENTBRICK IS NULL";
					else wtempstr = "ID_EVENTBRICK = " + i_ID_EVENTBRICK.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_PROCESSOPERATION != null)
				{
					if (i_ID_PROCESSOPERATION == System.DBNull.Value) wtempstr = "ID_PROCESSOPERATION IS NULL";
					else wtempstr = "ID_PROCESSOPERATION = " + i_ID_PROCESSOPERATION.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_PATH != null)
				{
					if (i_ID_PATH == System.DBNull.Value) wtempstr = "ID_PATH IS NULL";
					else wtempstr = "ID_PATH = " + i_ID_PATH.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (order == OrderBy.Ascending) return SelectWhere(wherestr, "ID_EVENTBRICK ASC,ID_PROCESSOPERATION ASC,ID_PATH ASC");
				else if (order == OrderBy.Descending) return SelectWhere(wherestr, "ID_EVENTBRICK DESC,ID_PROCESSOPERATION DESC,ID_PATH DESC");
				return SelectWhere(wherestr, null);
			}
			/// <summary>
			/// Reads a set of rows from TB_SCANBACK_CHECKRESULTS and retrieves them into a new TB_SCANBACK_CHECKRESULTS object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the TB_SCANBACK_CHECKRESULTS class that can be used to read the retrieved data.</returns>
			static public TB_SCANBACK_CHECKRESULTS SelectWhere(string wherestr, string orderstr)
			{
				TB_SCANBACK_CHECKRESULTS newobj = new TB_SCANBACK_CHECKRESULTS();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_EVENTBRICK,ID_PROCESSOPERATION,ID_PATH,ID_PLATE,RESULT FROM TB_SCANBACK_CHECKRESULTS" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO TB_SCANBACK_CHECKRESULTS (ID_EVENTBRICK,ID_PROCESSOPERATION,ID_PATH,ID_PLATE,RESULT) VALUES (:p_1,:p_2,:p_3,:p_4,:p_5)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_EVENTBRICK;
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_PROCESSOPERATION;
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_PATH;
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_PLATE;
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input).Value = a_RESULT;
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the TB_SCANBACK_PATHS table in the DB.
		/// For data insertion, the Insert method is used. Rows are inserted one by one.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class TB_SCANBACK_PATHS
		{
			internal TB_SCANBACK_PATHS() {}
			System.Data.DataRowCollection m_DRC;
			/// <summary>
			/// Retrieves ID_EVENTBRICK for the current row.
			/// </summary>
			public long _ID_EVENTBRICK
			{
				get
				{
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			/// <summary>
			/// Retrieves ID_PROCESSOPERATION for the current row.
			/// </summary>
			public long _ID_PROCESSOPERATION
			{
				get
				{
					return System.Convert.ToInt64(m_DR[1]);
				}
			}
			/// <summary>
			/// Retrieves PATH for the current row.
			/// </summary>
			public long _PATH
			{
				get
				{
					return System.Convert.ToInt64(m_DR[2]);
				}
			}
			/// <summary>
			/// Retrieves ID for the current row.
			/// </summary>
			public long _ID
			{
				get
				{
					return System.Convert.ToInt64(m_DR[3]);
				}
			}
			/// <summary>
			/// Retrieves ID_START_PLATE for the current row.
			/// </summary>
			public long _ID_START_PLATE
			{
				get
				{
					return System.Convert.ToInt64(m_DR[4]);
				}
			}
			/// <summary>
			/// Retrieves ID_FORK_PATH for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _ID_FORK_PATH
			{
				get
				{
					if (m_DR[5] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[5]);
				}
			}
			/// <summary>
			/// Retrieves ID_CANCEL_PLATE for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _ID_CANCEL_PLATE
			{
				get
				{
					if (m_DR[6] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[6]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is inserted immediately.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">the value to be inserted for ID_EVENTBRICK.</param>
			/// <param name="i_ID_PROCESSOPERATION">the value to be inserted for ID_PROCESSOPERATION.</param>
			/// <param name="i_PATH">the value to be inserted for PATH.</param>
			/// <param name="i_ID">the value to be inserted for ID. This value is actually used only if this method call is involved in data publication/replication, otherwise the actual value is generated by the OPERA DB and the supplied value is ignored.</param>
			/// <param name="i_ID_START_PLATE">the value to be inserted for ID_START_PLATE.</param>
			/// <param name="i_ID_FORK_PATH">the value to be inserted for ID_FORK_PATH. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_ID_CANCEL_PLATE">the value to be inserted for ID_CANCEL_PLATE. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <returns>the value of ID for the new row.</returns>
			static public long Insert(long i_ID_EVENTBRICK,long i_ID_PROCESSOPERATION,long i_PATH,long i_ID,long i_ID_START_PLATE,object i_ID_FORK_PATH,object i_ID_CANCEL_PLATE)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = i_ID_EVENTBRICK;
				cmd.Parameters[1].Value = i_ID_PROCESSOPERATION;
				cmd.Parameters[2].Value = i_PATH;
				cmd.Parameters[3].Value = i_ID;
				cmd.Parameters[4].Value = i_ID_START_PLATE;
				cmd.Parameters[5].Value = i_ID_FORK_PATH;
				cmd.Parameters[6].Value = i_ID_CANCEL_PLATE;
				cmd.ExecuteNonQuery();
				return SySal.OperaDb.Convert.ToInt64(cmd.Parameters[7].Value);
			}
			/// <summary>
			/// Reads a set of rows from TB_SCANBACK_PATHS and retrieves them into a new TB_SCANBACK_PATHS object.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_PROCESSOPERATION">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_PATH">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="order">the ordering scheme to be applied to returned rows. This applies to all columns in the primary key.</param>
			/// <returns>a new instance of the TB_SCANBACK_PATHS class that can be used to read the retrieved data.</returns>
			static public TB_SCANBACK_PATHS SelectPrimaryKey(object i_ID_EVENTBRICK,object i_ID_PROCESSOPERATION,object i_PATH, OrderBy order)
			{
				string wherestr = "";
				string wtempstr = "";
				if (i_ID_EVENTBRICK != null)
				{
					if (i_ID_EVENTBRICK == System.DBNull.Value) wtempstr = "ID_EVENTBRICK IS NULL";
					else wtempstr = "ID_EVENTBRICK = " + i_ID_EVENTBRICK.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_PROCESSOPERATION != null)
				{
					if (i_ID_PROCESSOPERATION == System.DBNull.Value) wtempstr = "ID_PROCESSOPERATION IS NULL";
					else wtempstr = "ID_PROCESSOPERATION = " + i_ID_PROCESSOPERATION.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_PATH != null)
				{
					if (i_PATH == System.DBNull.Value) wtempstr = "PATH IS NULL";
					else wtempstr = "PATH = " + i_PATH.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (order == OrderBy.Ascending) return SelectWhere(wherestr, "ID_EVENTBRICK ASC,ID_PROCESSOPERATION ASC,PATH ASC");
				else if (order == OrderBy.Descending) return SelectWhere(wherestr, "ID_EVENTBRICK DESC,ID_PROCESSOPERATION DESC,PATH DESC");
				return SelectWhere(wherestr, null);
			}
			/// <summary>
			/// Reads a set of rows from TB_SCANBACK_PATHS and retrieves them into a new TB_SCANBACK_PATHS object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the TB_SCANBACK_PATHS class that can be used to read the retrieved data.</returns>
			static public TB_SCANBACK_PATHS SelectWhere(string wherestr, string orderstr)
			{
				TB_SCANBACK_PATHS newobj = new TB_SCANBACK_PATHS();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_EVENTBRICK,ID_PROCESSOPERATION,PATH,ID,ID_START_PLATE,ID_FORK_PATH,ID_CANCEL_PLATE FROM TB_SCANBACK_PATHS" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO TB_SCANBACK_PATHS (ID_EVENTBRICK,ID_PROCESSOPERATION,PATH,ID,ID_START_PLATE,ID_FORK_PATH,ID_CANCEL_PLATE) VALUES (:p_1,:p_2,:p_3,:p_4,:p_5,:p_6,:p_7) RETURNING ID INTO :o_7");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_7", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("o_7", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Output);
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the TB_SCANBACK_PREDICTIONS table in the DB.
		/// For data insertion, the Insert and Flush static methods are used. This class inserts data in batches.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class TB_SCANBACK_PREDICTIONS
		{
			internal TB_SCANBACK_PREDICTIONS() {}
			System.Data.DataRowCollection m_DRC;
			const int ArraySize = 100;
			static int index = 0;
			/// <summary>
			/// Since this class uses batch insertion, this method must be called to ensure that all rows are actually inserted into TB_SCANBACK_PREDICTIONS. Failure to do so will result in incomplete writes.
			/// </summary>
			static public void Flush() { if (cmd == null || index == 0) return; cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			private static long [] a_ID_EVENTBRICK = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_EVENTBRICK for the current row.
			/// </summary>
			public long _ID_EVENTBRICK
			{
				get
				{
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			private static long [] a_ID_PATH = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_PATH for the current row.
			/// </summary>
			public long _ID_PATH
			{
				get
				{
					return System.Convert.ToInt64(m_DR[1]);
				}
			}
			private static long [] a_ID_PLATE = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_PLATE for the current row.
			/// </summary>
			public long _ID_PLATE
			{
				get
				{
					return System.Convert.ToInt64(m_DR[2]);
				}
			}
			private static double [] a_POSX = new double[ArraySize];
			/// <summary>
			/// Retrieves POSX for the current row.
			/// </summary>
			public double _POSX
			{
				get
				{
					return System.Convert.ToDouble(m_DR[3]);
				}
			}
			private static double [] a_POSY = new double[ArraySize];
			/// <summary>
			/// Retrieves POSY for the current row.
			/// </summary>
			public double _POSY
			{
				get
				{
					return System.Convert.ToDouble(m_DR[4]);
				}
			}
			private static double [] a_SLOPEX = new double[ArraySize];
			/// <summary>
			/// Retrieves SLOPEX for the current row.
			/// </summary>
			public double _SLOPEX
			{
				get
				{
					return System.Convert.ToDouble(m_DR[5]);
				}
			}
			private static double [] a_SLOPEY = new double[ArraySize];
			/// <summary>
			/// Retrieves SLOPEY for the current row.
			/// </summary>
			public double _SLOPEY
			{
				get
				{
					return System.Convert.ToDouble(m_DR[6]);
				}
			}
			private static object [] a_POSTOL1 = new object[ArraySize];
			/// <summary>
			/// Retrieves POSTOL1 for the current row. The return value can be System.DBNull.Value or a value that can be cast to double.
			/// </summary>
			public object _POSTOL1
			{
				get
				{
					if (m_DR[7] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDouble(m_DR[7]);
				}
			}
			private static object [] a_POSTOL2 = new object[ArraySize];
			/// <summary>
			/// Retrieves POSTOL2 for the current row. The return value can be System.DBNull.Value or a value that can be cast to double.
			/// </summary>
			public object _POSTOL2
			{
				get
				{
					if (m_DR[8] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDouble(m_DR[8]);
				}
			}
			private static object [] a_SLOPETOL1 = new object[ArraySize];
			/// <summary>
			/// Retrieves SLOPETOL1 for the current row. The return value can be System.DBNull.Value or a value that can be cast to double.
			/// </summary>
			public object _SLOPETOL1
			{
				get
				{
					if (m_DR[9] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDouble(m_DR[9]);
				}
			}
			private static object [] a_SLOPETOL2 = new object[ArraySize];
			/// <summary>
			/// Retrieves SLOPETOL2 for the current row. The return value can be System.DBNull.Value or a value that can be cast to double.
			/// </summary>
			public object _SLOPETOL2
			{
				get
				{
					if (m_DR[10] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDouble(m_DR[10]);
				}
			}
			private static object [] a_FRAME = new object[ArraySize];
			/// <summary>
			/// Retrieves FRAME for the current row. The return value can be System.DBNull.Value or a value that can be cast to char.
			/// </summary>
			public object _FRAME
			{
				get
				{
					if (m_DR[11] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToChar(m_DR[11]);
				}
			}
			private static object [] a_ID_ZONE = new object[ArraySize];
			/// <summary>
			/// Retrieves ID_ZONE for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _ID_ZONE
			{
				get
				{
					if (m_DR[12] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[12]);
				}
			}
			private static object [] a_ID_CANDIDATE = new object[ArraySize];
			/// <summary>
			/// Retrieves ID_CANDIDATE for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _ID_CANDIDATE
			{
				get
				{
					if (m_DR[13] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[13]);
				}
			}
			private static object [] a_DAMAGED = new object[ArraySize];
			/// <summary>
			/// Retrieves DAMAGED for the current row. The return value can be System.DBNull.Value or a value that can be cast to char.
			/// </summary>
			public object _DAMAGED
			{
				get
				{
					if (m_DR[14] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToChar(m_DR[14]);
				}
			}
			private static object [] a_ISMANUAL = new object[ArraySize];
			/// <summary>
			/// Retrieves ISMANUAL for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _ISMANUAL
			{
				get
				{
					if (m_DR[15] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[15]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is actually inserted later, in a batch insertion command. The Flush method must be called to ensure all rows are actually written.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">the value to be inserted for ID_EVENTBRICK.</param>
			/// <param name="i_ID_PATH">the value to be inserted for ID_PATH.</param>
			/// <param name="i_ID_PLATE">the value to be inserted for ID_PLATE.</param>
			/// <param name="i_POSX">the value to be inserted for POSX.</param>
			/// <param name="i_POSY">the value to be inserted for POSY.</param>
			/// <param name="i_SLOPEX">the value to be inserted for SLOPEX.</param>
			/// <param name="i_SLOPEY">the value to be inserted for SLOPEY.</param>
			/// <param name="i_POSTOL1">the value to be inserted for POSTOL1. The value for this parameter can be double or System.DBNull.Value.</param>
			/// <param name="i_POSTOL2">the value to be inserted for POSTOL2. The value for this parameter can be double or System.DBNull.Value.</param>
			/// <param name="i_SLOPETOL1">the value to be inserted for SLOPETOL1. The value for this parameter can be double or System.DBNull.Value.</param>
			/// <param name="i_SLOPETOL2">the value to be inserted for SLOPETOL2. The value for this parameter can be double or System.DBNull.Value.</param>
			/// <param name="i_FRAME">the value to be inserted for FRAME. The value for this parameter can be char or System.DBNull.Value.</param>
			/// <param name="i_ID_ZONE">the value to be inserted for ID_ZONE. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_ID_CANDIDATE">the value to be inserted for ID_CANDIDATE. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_DAMAGED">the value to be inserted for DAMAGED. The value for this parameter can be char or System.DBNull.Value.</param>
			/// <param name="i_ISMANUAL">the value to be inserted for ISMANUAL. The value for this parameter can be long or System.DBNull.Value.</param>
			static public void Insert(long i_ID_EVENTBRICK,long i_ID_PATH,long i_ID_PLATE,double i_POSX,double i_POSY,double i_SLOPEX,double i_SLOPEY,object i_POSTOL1,object i_POSTOL2,object i_SLOPETOL1,object i_SLOPETOL2,object i_FRAME,object i_ID_ZONE,object i_ID_CANDIDATE,object i_DAMAGED,object i_ISMANUAL)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				a_ID_EVENTBRICK[index] = i_ID_EVENTBRICK;
				a_ID_PATH[index] = i_ID_PATH;
				a_ID_PLATE[index] = i_ID_PLATE;
				a_POSX[index] = i_POSX;
				a_POSY[index] = i_POSY;
				a_SLOPEX[index] = i_SLOPEX;
				a_SLOPEY[index] = i_SLOPEY;
				a_POSTOL1[index] = i_POSTOL1;
				a_POSTOL2[index] = i_POSTOL2;
				a_SLOPETOL1[index] = i_SLOPETOL1;
				a_SLOPETOL2[index] = i_SLOPETOL2;
				a_FRAME[index] = i_FRAME;
				a_ID_ZONE[index] = i_ID_ZONE;
				a_ID_CANDIDATE[index] = i_ID_CANDIDATE;
				a_DAMAGED[index] = i_DAMAGED;
				a_ISMANUAL[index] = i_ISMANUAL;
				if (++index >= ArraySize) { cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			}
			/// <summary>
			/// Reads a set of rows from TB_SCANBACK_PREDICTIONS and retrieves them into a new TB_SCANBACK_PREDICTIONS object.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_PATH">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_PLATE">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="order">the ordering scheme to be applied to returned rows. This applies to all columns in the primary key.</param>
			/// <returns>a new instance of the TB_SCANBACK_PREDICTIONS class that can be used to read the retrieved data.</returns>
			static public TB_SCANBACK_PREDICTIONS SelectPrimaryKey(object i_ID_EVENTBRICK,object i_ID_PATH,object i_ID_PLATE, OrderBy order)
			{
				string wherestr = "";
				string wtempstr = "";
				if (i_ID_EVENTBRICK != null)
				{
					if (i_ID_EVENTBRICK == System.DBNull.Value) wtempstr = "ID_EVENTBRICK IS NULL";
					else wtempstr = "ID_EVENTBRICK = " + i_ID_EVENTBRICK.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_PATH != null)
				{
					if (i_ID_PATH == System.DBNull.Value) wtempstr = "ID_PATH IS NULL";
					else wtempstr = "ID_PATH = " + i_ID_PATH.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_PLATE != null)
				{
					if (i_ID_PLATE == System.DBNull.Value) wtempstr = "ID_PLATE IS NULL";
					else wtempstr = "ID_PLATE = " + i_ID_PLATE.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (order == OrderBy.Ascending) return SelectWhere(wherestr, "ID_EVENTBRICK ASC,ID_PATH ASC,ID_PLATE ASC");
				else if (order == OrderBy.Descending) return SelectWhere(wherestr, "ID_EVENTBRICK DESC,ID_PATH DESC,ID_PLATE DESC");
				return SelectWhere(wherestr, null);
			}
			/// <summary>
			/// Reads a set of rows from TB_SCANBACK_PREDICTIONS and retrieves them into a new TB_SCANBACK_PREDICTIONS object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the TB_SCANBACK_PREDICTIONS class that can be used to read the retrieved data.</returns>
			static public TB_SCANBACK_PREDICTIONS SelectWhere(string wherestr, string orderstr)
			{
				TB_SCANBACK_PREDICTIONS newobj = new TB_SCANBACK_PREDICTIONS();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_EVENTBRICK,ID_PATH,ID_PLATE,POSX,POSY,SLOPEX,SLOPEY,POSTOL1,POSTOL2,SLOPETOL1,SLOPETOL2,FRAME,ID_ZONE,ID_CANDIDATE,DAMAGED,ISMANUAL FROM TB_SCANBACK_PREDICTIONS" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO TB_SCANBACK_PREDICTIONS (ID_EVENTBRICK,ID_PATH,ID_PLATE,POSX,POSY,SLOPEX,SLOPEY,POSTOL1,POSTOL2,SLOPETOL1,SLOPETOL2,FRAME,ID_ZONE,ID_CANDIDATE,DAMAGED,ISMANUAL) VALUES (:p_1,:p_2,:p_3,:p_4,:p_5,:p_6,:p_7,:p_8,:p_9,:p_10,:p_11,:p_12,:p_13,:p_14,:p_15,:p_16)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_EVENTBRICK;
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_PATH;
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_PLATE;
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_POSX;
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_POSY;
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_SLOPEX;
				newcmd.Parameters.Add("p_7", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_SLOPEY;
				newcmd.Parameters.Add("p_8", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_POSTOL1;
				newcmd.Parameters.Add("p_9", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_POSTOL2;
				newcmd.Parameters.Add("p_10", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_SLOPETOL1;
				newcmd.Parameters.Add("p_11", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_SLOPETOL2;
				newcmd.Parameters.Add("p_12", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input).Value = a_FRAME;
				newcmd.Parameters.Add("p_13", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_ZONE;
				newcmd.Parameters.Add("p_14", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_CANDIDATE;
				newcmd.Parameters.Add("p_15", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input).Value = a_DAMAGED;
				newcmd.Parameters.Add("p_16", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ISMANUAL;
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the TB_SITES table in the DB.
		/// For data insertion, the Insert method is used. Rows are inserted one by one.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class TB_SITES
		{
			internal TB_SITES() {}
			System.Data.DataRowCollection m_DRC;
			/// <summary>
			/// Retrieves ID for the current row.
			/// </summary>
			public long _ID
			{
				get
				{
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			/// <summary>
			/// Retrieves NAME for the current row.
			/// </summary>
			public string _NAME
			{
				get
				{
					return System.Convert.ToString(m_DR[1]);
				}
			}
			/// <summary>
			/// Retrieves LATITUDE for the current row.
			/// </summary>
			public double _LATITUDE
			{
				get
				{
					return System.Convert.ToDouble(m_DR[2]);
				}
			}
			/// <summary>
			/// Retrieves LONGITUDE for the current row.
			/// </summary>
			public double _LONGITUDE
			{
				get
				{
					return System.Convert.ToDouble(m_DR[3]);
				}
			}
			/// <summary>
			/// Retrieves LOCALTIMEFUSE for the current row.
			/// </summary>
			public int _LOCALTIMEFUSE
			{
				get
				{
					return System.Convert.ToInt32(m_DR[4]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is inserted immediately.
			/// </summary>
			/// <param name="i_ID">the value to be inserted for ID. This value is actually used only if this method call is involved in data publication/replication, otherwise the actual value is generated by the OPERA DB and the supplied value is ignored.</param>
			/// <param name="i_NAME">the value to be inserted for NAME.</param>
			/// <param name="i_LATITUDE">the value to be inserted for LATITUDE.</param>
			/// <param name="i_LONGITUDE">the value to be inserted for LONGITUDE.</param>
			/// <param name="i_LOCALTIMEFUSE">the value to be inserted for LOCALTIMEFUSE.</param>
			/// <returns>the value of ID for the new row.</returns>
			static public long Insert(long i_ID,string i_NAME,double i_LATITUDE,double i_LONGITUDE,int i_LOCALTIMEFUSE)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = i_ID;
				cmd.Parameters[1].Value = i_NAME;
				cmd.Parameters[2].Value = i_LATITUDE;
				cmd.Parameters[3].Value = i_LONGITUDE;
				cmd.Parameters[4].Value = i_LOCALTIMEFUSE;
				cmd.ExecuteNonQuery();
				return SySal.OperaDb.Convert.ToInt64(cmd.Parameters[5].Value);
			}
			/// <summary>
			/// Reads a set of rows from TB_SITES and retrieves them into a new TB_SITES object.
			/// </summary>
			/// <param name="i_NAME">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="order">the ordering scheme to be applied to returned rows.</param>
			/// <returns>a new instance of the TB_SITES class that can be used to read the retrieved data.</returns>
			static public TB_SITES SelectPrimaryKey(object i_NAME, OrderBy order)
			{
				string wherestr = "";
				string wtempstr = "";
				if (i_NAME != null)
				{
					if (i_NAME == System.DBNull.Value) wtempstr = "NAME IS NULL";
					else wtempstr = "NAME = " + i_NAME.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (order == OrderBy.Ascending) return SelectWhere(wherestr, "NAME ASC");
				else if (order == OrderBy.Descending) return SelectWhere(wherestr, "NAME DESC");
				return SelectWhere(wherestr, null);
			}
			/// <summary>
			/// Reads a set of rows from TB_SITES and retrieves them into a new TB_SITES object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the TB_SITES class that can be used to read the retrieved data.</returns>
			static public TB_SITES SelectWhere(string wherestr, string orderstr)
			{
				TB_SITES newobj = new TB_SITES();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID,NAME,LATITUDE,LONGITUDE,LOCALTIMEFUSE FROM TB_SITES" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO TB_SITES (ID,NAME,LATITUDE,LONGITUDE,LOCALTIMEFUSE) VALUES (:p_1,:p_2,:p_3,:p_4,:p_5) RETURNING ID INTO :o_5");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Int, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("o_5", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Output);
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the TB_TEMPLATEMARKSETS table in the DB.
		/// For data insertion, the Insert method is used. Rows are inserted one by one.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class TB_TEMPLATEMARKSETS
		{
			internal TB_TEMPLATEMARKSETS() {}
			System.Data.DataRowCollection m_DRC;
			/// <summary>
			/// Retrieves ID for the current row.
			/// </summary>
			public long _ID
			{
				get
				{
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			/// <summary>
			/// Retrieves ID_EVENTBRICK for the current row.
			/// </summary>
			public long _ID_EVENTBRICK
			{
				get
				{
					return System.Convert.ToInt64(m_DR[1]);
				}
			}
			/// <summary>
			/// Retrieves ID_MARK for the current row.
			/// </summary>
			public long _ID_MARK
			{
				get
				{
					return System.Convert.ToInt64(m_DR[2]);
				}
			}
			/// <summary>
			/// Retrieves POSX for the current row.
			/// </summary>
			public double _POSX
			{
				get
				{
					return System.Convert.ToDouble(m_DR[3]);
				}
			}
			/// <summary>
			/// Retrieves POSY for the current row.
			/// </summary>
			public double _POSY
			{
				get
				{
					return System.Convert.ToDouble(m_DR[4]);
				}
			}
			/// <summary>
			/// Retrieves MARKROW for the current row.
			/// </summary>
			public int _MARKROW
			{
				get
				{
					return System.Convert.ToInt32(m_DR[5]);
				}
			}
			/// <summary>
			/// Retrieves MARKCOL for the current row.
			/// </summary>
			public int _MARKCOL
			{
				get
				{
					return System.Convert.ToInt32(m_DR[6]);
				}
			}
			/// <summary>
			/// Retrieves SHAPE for the current row.
			/// </summary>
			public string _SHAPE
			{
				get
				{
					return System.Convert.ToString(m_DR[7]);
				}
			}
			/// <summary>
			/// Retrieves SIDE for the current row.
			/// </summary>
			public int _SIDE
			{
				get
				{
					return System.Convert.ToInt32(m_DR[8]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is inserted immediately.
			/// </summary>
			/// <param name="i_ID">the value to be inserted for ID. This value is actually used only if this method call is involved in data publication/replication, otherwise the actual value is generated by the OPERA DB and the supplied value is ignored.</param>
			/// <param name="i_ID_EVENTBRICK">the value to be inserted for ID_EVENTBRICK.</param>
			/// <param name="i_ID_MARK">the value to be inserted for ID_MARK.</param>
			/// <param name="i_POSX">the value to be inserted for POSX.</param>
			/// <param name="i_POSY">the value to be inserted for POSY.</param>
			/// <param name="i_MARKROW">the value to be inserted for MARKROW.</param>
			/// <param name="i_MARKCOL">the value to be inserted for MARKCOL.</param>
			/// <param name="i_SHAPE">the value to be inserted for SHAPE.</param>
			/// <param name="i_SIDE">the value to be inserted for SIDE.</param>
			/// <returns>the value of ID for the new row.</returns>
			static public long Insert(long i_ID,long i_ID_EVENTBRICK,long i_ID_MARK,double i_POSX,double i_POSY,int i_MARKROW,int i_MARKCOL,string i_SHAPE,int i_SIDE)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = i_ID;
				cmd.Parameters[1].Value = i_ID_EVENTBRICK;
				cmd.Parameters[2].Value = i_ID_MARK;
				cmd.Parameters[3].Value = i_POSX;
				cmd.Parameters[4].Value = i_POSY;
				cmd.Parameters[5].Value = i_MARKROW;
				cmd.Parameters[6].Value = i_MARKCOL;
				cmd.Parameters[7].Value = i_SHAPE;
				cmd.Parameters[8].Value = i_SIDE;
				cmd.ExecuteNonQuery();
				return SySal.OperaDb.Convert.ToInt64(cmd.Parameters[9].Value);
			}
			/// <summary>
			/// Reads a set of rows from TB_TEMPLATEMARKSETS and retrieves them into a new TB_TEMPLATEMARKSETS object.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_MARK">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_MARKROW">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_MARKCOL">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="order">the ordering scheme to be applied to returned rows. This applies to all columns in the primary key.</param>
			/// <returns>a new instance of the TB_TEMPLATEMARKSETS class that can be used to read the retrieved data.</returns>
			static public TB_TEMPLATEMARKSETS SelectPrimaryKey(object i_ID_EVENTBRICK,object i_ID_MARK,object i_MARKROW,object i_MARKCOL, OrderBy order)
			{
				string wherestr = "";
				string wtempstr = "";
				if (i_ID_EVENTBRICK != null)
				{
					if (i_ID_EVENTBRICK == System.DBNull.Value) wtempstr = "ID_EVENTBRICK IS NULL";
					else wtempstr = "ID_EVENTBRICK = " + i_ID_EVENTBRICK.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_MARK != null)
				{
					if (i_ID_MARK == System.DBNull.Value) wtempstr = "ID_MARK IS NULL";
					else wtempstr = "ID_MARK = " + i_ID_MARK.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_MARKROW != null)
				{
					if (i_MARKROW == System.DBNull.Value) wtempstr = "MARKROW IS NULL";
					else wtempstr = "MARKROW = " + i_MARKROW.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_MARKCOL != null)
				{
					if (i_MARKCOL == System.DBNull.Value) wtempstr = "MARKCOL IS NULL";
					else wtempstr = "MARKCOL = " + i_MARKCOL.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (order == OrderBy.Ascending) return SelectWhere(wherestr, "ID_EVENTBRICK ASC,ID_MARK ASC,MARKROW ASC,MARKCOL ASC");
				else if (order == OrderBy.Descending) return SelectWhere(wherestr, "ID_EVENTBRICK DESC,ID_MARK DESC,MARKROW DESC,MARKCOL DESC");
				return SelectWhere(wherestr, null);
			}
			/// <summary>
			/// Reads a set of rows from TB_TEMPLATEMARKSETS and retrieves them into a new TB_TEMPLATEMARKSETS object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the TB_TEMPLATEMARKSETS class that can be used to read the retrieved data.</returns>
			static public TB_TEMPLATEMARKSETS SelectWhere(string wherestr, string orderstr)
			{
				TB_TEMPLATEMARKSETS newobj = new TB_TEMPLATEMARKSETS();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID,ID_EVENTBRICK,ID_MARK,POSX,POSY,MARKROW,MARKCOL,SHAPE,SIDE FROM TB_TEMPLATEMARKSETS" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO TB_TEMPLATEMARKSETS (ID,ID_EVENTBRICK,ID_MARK,POSX,POSY,MARKROW,MARKCOL,SHAPE,SIDE) VALUES (:p_1,:p_2,:p_3,:p_4,:p_5,:p_6,:p_7,:p_8,:p_9) RETURNING ID INTO :o_9");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.Int, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_7", SySal.OperaDb.OperaDbType.Int, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_8", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_9", SySal.OperaDb.OperaDbType.Int, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("o_9", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Output);
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the TB_USERS table in the DB.
		/// For data insertion, the Insert method is used. Rows are inserted one by one.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class TB_USERS
		{
			internal TB_USERS() {}
			System.Data.DataRowCollection m_DRC;
			/// <summary>
			/// Retrieves ID for the current row.
			/// </summary>
			public long _ID
			{
				get
				{
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			/// <summary>
			/// Retrieves USERNAME for the current row.
			/// </summary>
			public string _USERNAME
			{
				get
				{
					return System.Convert.ToString(m_DR[1]);
				}
			}
			/// <summary>
			/// Retrieves PWD for the current row.
			/// </summary>
			public string _PWD
			{
				get
				{
					return System.Convert.ToString(m_DR[2]);
				}
			}
			/// <summary>
			/// Retrieves NAME for the current row.
			/// </summary>
			public string _NAME
			{
				get
				{
					return System.Convert.ToString(m_DR[3]);
				}
			}
			/// <summary>
			/// Retrieves SURNAME for the current row.
			/// </summary>
			public string _SURNAME
			{
				get
				{
					return System.Convert.ToString(m_DR[4]);
				}
			}
			/// <summary>
			/// Retrieves INSTITUTION for the current row.
			/// </summary>
			public string _INSTITUTION
			{
				get
				{
					return System.Convert.ToString(m_DR[5]);
				}
			}
			/// <summary>
			/// Retrieves ID_SITE for the current row.
			/// </summary>
			public long _ID_SITE
			{
				get
				{
					return System.Convert.ToInt64(m_DR[6]);
				}
			}
			/// <summary>
			/// Retrieves EMAIL for the current row.
			/// </summary>
			public string _EMAIL
			{
				get
				{
					return System.Convert.ToString(m_DR[7]);
				}
			}
			/// <summary>
			/// Retrieves ADDRESS for the current row.
			/// </summary>
			public string _ADDRESS
			{
				get
				{
					return System.Convert.ToString(m_DR[8]);
				}
			}
			/// <summary>
			/// Retrieves PHONE for the current row.
			/// </summary>
			public string _PHONE
			{
				get
				{
					return System.Convert.ToString(m_DR[9]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is inserted immediately.
			/// </summary>
			/// <param name="i_ID">the value to be inserted for ID. This value is actually used only if this method call is involved in data publication/replication, otherwise the actual value is generated by the OPERA DB and the supplied value is ignored.</param>
			/// <param name="i_USERNAME">the value to be inserted for USERNAME.</param>
			/// <param name="i_PWD">the value to be inserted for PWD.</param>
			/// <param name="i_NAME">the value to be inserted for NAME.</param>
			/// <param name="i_SURNAME">the value to be inserted for SURNAME.</param>
			/// <param name="i_INSTITUTION">the value to be inserted for INSTITUTION.</param>
			/// <param name="i_ID_SITE">the value to be inserted for ID_SITE.</param>
			/// <param name="i_EMAIL">the value to be inserted for EMAIL.</param>
			/// <param name="i_ADDRESS">the value to be inserted for ADDRESS.</param>
			/// <param name="i_PHONE">the value to be inserted for PHONE.</param>
			/// <returns>the value of ID for the new row.</returns>
			static public long Insert(long i_ID,string i_USERNAME,string i_PWD,string i_NAME,string i_SURNAME,string i_INSTITUTION,long i_ID_SITE,string i_EMAIL,string i_ADDRESS,string i_PHONE)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = i_ID;
				cmd.Parameters[1].Value = i_USERNAME;
				cmd.Parameters[2].Value = i_PWD;
				cmd.Parameters[3].Value = i_NAME;
				cmd.Parameters[4].Value = i_SURNAME;
				cmd.Parameters[5].Value = i_INSTITUTION;
				cmd.Parameters[6].Value = i_ID_SITE;
				cmd.Parameters[7].Value = i_EMAIL;
				cmd.Parameters[8].Value = i_ADDRESS;
				cmd.Parameters[9].Value = i_PHONE;
				cmd.ExecuteNonQuery();
				return SySal.OperaDb.Convert.ToInt64(cmd.Parameters[10].Value);
			}
			/// <summary>
			/// Reads a set of rows from TB_USERS and retrieves them into a new TB_USERS object.
			/// </summary>
			/// <param name="i_NAME">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_SURNAME">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="order">the ordering scheme to be applied to returned rows. This applies to all columns in the primary key.</param>
			/// <returns>a new instance of the TB_USERS class that can be used to read the retrieved data.</returns>
			static public TB_USERS SelectPrimaryKey(object i_NAME,object i_SURNAME, OrderBy order)
			{
				string wherestr = "";
				string wtempstr = "";
				if (i_NAME != null)
				{
					if (i_NAME == System.DBNull.Value) wtempstr = "NAME IS NULL";
					else wtempstr = "NAME = " + i_NAME.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_SURNAME != null)
				{
					if (i_SURNAME == System.DBNull.Value) wtempstr = "SURNAME IS NULL";
					else wtempstr = "SURNAME = " + i_SURNAME.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (order == OrderBy.Ascending) return SelectWhere(wherestr, "NAME ASC,SURNAME ASC");
				else if (order == OrderBy.Descending) return SelectWhere(wherestr, "NAME DESC,SURNAME DESC");
				return SelectWhere(wherestr, null);
			}
			/// <summary>
			/// Reads a set of rows from TB_USERS and retrieves them into a new TB_USERS object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the TB_USERS class that can be used to read the retrieved data.</returns>
			static public TB_USERS SelectWhere(string wherestr, string orderstr)
			{
				TB_USERS newobj = new TB_USERS();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID,USERNAME,PWD,NAME,SURNAME,INSTITUTION,ID_SITE,EMAIL,ADDRESS,PHONE FROM TB_USERS" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO TB_USERS (ID,USERNAME,PWD,NAME,SURNAME,INSTITUTION,ID_SITE,EMAIL,ADDRESS,PHONE) VALUES (:p_1,:p_2,:p_3,:p_4,:p_5,:p_6,:p_7,:p_8,:p_9,:p_10) RETURNING ID INTO :o_10");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_7", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_8", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_9", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_10", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("o_10", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Output);
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the TB_VERTICES table in the DB.
		/// For data insertion, the Insert and Flush static methods are used. This class inserts data in batches.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class TB_VERTICES
		{
			internal TB_VERTICES() {}
			System.Data.DataRowCollection m_DRC;
			const int ArraySize = 100;
			static int index = 0;
			/// <summary>
			/// Since this class uses batch insertion, this method must be called to ensure that all rows are actually inserted into TB_VERTICES. Failure to do so will result in incomplete writes.
			/// </summary>
			static public void Flush() { if (cmd == null || index == 0) return; cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			private static long [] a_ID_EVENTBRICK = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_EVENTBRICK for the current row.
			/// </summary>
			public long _ID_EVENTBRICK
			{
				get
				{
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			private static long [] a_ID_RECONSTRUCTION = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_RECONSTRUCTION for the current row.
			/// </summary>
			public long _ID_RECONSTRUCTION
			{
				get
				{
					return System.Convert.ToInt64(m_DR[1]);
				}
			}
			private static long [] a_ID_OPTION = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_OPTION for the current row.
			/// </summary>
			public long _ID_OPTION
			{
				get
				{
					return System.Convert.ToInt64(m_DR[2]);
				}
			}
			private static long [] a_ID = new long[ArraySize];
			/// <summary>
			/// Retrieves ID for the current row.
			/// </summary>
			public long _ID
			{
				get
				{
					return System.Convert.ToInt64(m_DR[3]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is actually inserted later, in a batch insertion command. The Flush method must be called to ensure all rows are actually written.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">the value to be inserted for ID_EVENTBRICK.</param>
			/// <param name="i_ID_RECONSTRUCTION">the value to be inserted for ID_RECONSTRUCTION.</param>
			/// <param name="i_ID_OPTION">the value to be inserted for ID_OPTION.</param>
			/// <param name="i_ID">the value to be inserted for ID.</param>
			static public void Insert(long i_ID_EVENTBRICK,long i_ID_RECONSTRUCTION,long i_ID_OPTION,long i_ID)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				a_ID_EVENTBRICK[index] = i_ID_EVENTBRICK;
				a_ID_RECONSTRUCTION[index] = i_ID_RECONSTRUCTION;
				a_ID_OPTION[index] = i_ID_OPTION;
				a_ID[index] = i_ID;
				if (++index >= ArraySize) { cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			}
			/// <summary>
			/// Reads a set of rows from TB_VERTICES and retrieves them into a new TB_VERTICES object.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_RECONSTRUCTION">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_OPTION">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="order">the ordering scheme to be applied to returned rows. This applies to all columns in the primary key.</param>
			/// <returns>a new instance of the TB_VERTICES class that can be used to read the retrieved data.</returns>
			static public TB_VERTICES SelectPrimaryKey(object i_ID_EVENTBRICK,object i_ID_RECONSTRUCTION,object i_ID_OPTION,object i_ID, OrderBy order)
			{
				string wherestr = "";
				string wtempstr = "";
				if (i_ID_EVENTBRICK != null)
				{
					if (i_ID_EVENTBRICK == System.DBNull.Value) wtempstr = "ID_EVENTBRICK IS NULL";
					else wtempstr = "ID_EVENTBRICK = " + i_ID_EVENTBRICK.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_RECONSTRUCTION != null)
				{
					if (i_ID_RECONSTRUCTION == System.DBNull.Value) wtempstr = "ID_RECONSTRUCTION IS NULL";
					else wtempstr = "ID_RECONSTRUCTION = " + i_ID_RECONSTRUCTION.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_OPTION != null)
				{
					if (i_ID_OPTION == System.DBNull.Value) wtempstr = "ID_OPTION IS NULL";
					else wtempstr = "ID_OPTION = " + i_ID_OPTION.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID != null)
				{
					if (i_ID == System.DBNull.Value) wtempstr = "ID IS NULL";
					else wtempstr = "ID = " + i_ID.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (order == OrderBy.Ascending) return SelectWhere(wherestr, "ID_EVENTBRICK ASC,ID_RECONSTRUCTION ASC,ID_OPTION ASC,ID ASC");
				else if (order == OrderBy.Descending) return SelectWhere(wherestr, "ID_EVENTBRICK DESC,ID_RECONSTRUCTION DESC,ID_OPTION DESC,ID DESC");
				return SelectWhere(wherestr, null);
			}
			/// <summary>
			/// Reads a set of rows from TB_VERTICES and retrieves them into a new TB_VERTICES object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the TB_VERTICES class that can be used to read the retrieved data.</returns>
			static public TB_VERTICES SelectWhere(string wherestr, string orderstr)
			{
				TB_VERTICES newobj = new TB_VERTICES();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_EVENTBRICK,ID_RECONSTRUCTION,ID_OPTION,ID FROM TB_VERTICES" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO TB_VERTICES (ID_EVENTBRICK,ID_RECONSTRUCTION,ID_OPTION,ID) VALUES (:p_1,:p_2,:p_3,:p_4)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_EVENTBRICK;
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_RECONSTRUCTION;
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_OPTION;
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID;
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the TB_VERTICES_ATTR table in the DB.
		/// For data insertion, the Insert and Flush static methods are used. This class inserts data in batches.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class TB_VERTICES_ATTR
		{
			internal TB_VERTICES_ATTR() {}
			System.Data.DataRowCollection m_DRC;
			const int ArraySize = 100;
			static int index = 0;
			/// <summary>
			/// Since this class uses batch insertion, this method must be called to ensure that all rows are actually inserted into TB_VERTICES_ATTR. Failure to do so will result in incomplete writes.
			/// </summary>
			static public void Flush() { if (cmd == null || index == 0) return; cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			private static long [] a_ID_EVENTBRICK = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_EVENTBRICK for the current row.
			/// </summary>
			public long _ID_EVENTBRICK
			{
				get
				{
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			private static long [] a_ID_RECONSTRUCTION = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_RECONSTRUCTION for the current row.
			/// </summary>
			public long _ID_RECONSTRUCTION
			{
				get
				{
					return System.Convert.ToInt64(m_DR[1]);
				}
			}
			private static long [] a_ID_OPTION = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_OPTION for the current row.
			/// </summary>
			public long _ID_OPTION
			{
				get
				{
					return System.Convert.ToInt64(m_DR[2]);
				}
			}
			private static long [] a_ID = new long[ArraySize];
			/// <summary>
			/// Retrieves ID for the current row.
			/// </summary>
			public long _ID
			{
				get
				{
					return System.Convert.ToInt64(m_DR[3]);
				}
			}
			private static long [] a_ID_PROCESSOPERATION = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_PROCESSOPERATION for the current row.
			/// </summary>
			public long _ID_PROCESSOPERATION
			{
				get
				{
					return System.Convert.ToInt64(m_DR[4]);
				}
			}
			private static string [] a_NAME = new string[ArraySize];
			/// <summary>
			/// Retrieves NAME for the current row.
			/// </summary>
			public string _NAME
			{
				get
				{
					return System.Convert.ToString(m_DR[5]);
				}
			}
			private static object [] a_VALUE = new object[ArraySize];
			/// <summary>
			/// Retrieves VALUE for the current row. The return value can be System.DBNull.Value or a value that can be cast to double.
			/// </summary>
			public object _VALUE
			{
				get
				{
					if (m_DR[6] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDouble(m_DR[6]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is actually inserted later, in a batch insertion command. The Flush method must be called to ensure all rows are actually written.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">the value to be inserted for ID_EVENTBRICK.</param>
			/// <param name="i_ID_RECONSTRUCTION">the value to be inserted for ID_RECONSTRUCTION.</param>
			/// <param name="i_ID_OPTION">the value to be inserted for ID_OPTION.</param>
			/// <param name="i_ID">the value to be inserted for ID.</param>
			/// <param name="i_ID_PROCESSOPERATION">the value to be inserted for ID_PROCESSOPERATION.</param>
			/// <param name="i_NAME">the value to be inserted for NAME.</param>
			/// <param name="i_VALUE">the value to be inserted for VALUE. The value for this parameter can be double or System.DBNull.Value.</param>
			static public void Insert(long i_ID_EVENTBRICK,long i_ID_RECONSTRUCTION,long i_ID_OPTION,long i_ID,long i_ID_PROCESSOPERATION,string i_NAME,object i_VALUE)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				a_ID_EVENTBRICK[index] = i_ID_EVENTBRICK;
				a_ID_RECONSTRUCTION[index] = i_ID_RECONSTRUCTION;
				a_ID_OPTION[index] = i_ID_OPTION;
				a_ID[index] = i_ID;
				a_ID_PROCESSOPERATION[index] = i_ID_PROCESSOPERATION;
				a_NAME[index] = i_NAME;
				a_VALUE[index] = i_VALUE;
				if (++index >= ArraySize) { cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			}
			/// <summary>
			/// Reads a set of rows from TB_VERTICES_ATTR and retrieves them into a new TB_VERTICES_ATTR object.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_RECONSTRUCTION">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_OPTION">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_PROCESSOPERATION">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_NAME">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="order">the ordering scheme to be applied to returned rows. This applies to all columns in the primary key.</param>
			/// <returns>a new instance of the TB_VERTICES_ATTR class that can be used to read the retrieved data.</returns>
			static public TB_VERTICES_ATTR SelectPrimaryKey(object i_ID_EVENTBRICK,object i_ID_RECONSTRUCTION,object i_ID_OPTION,object i_ID,object i_ID_PROCESSOPERATION,object i_NAME, OrderBy order)
			{
				string wherestr = "";
				string wtempstr = "";
				if (i_ID_EVENTBRICK != null)
				{
					if (i_ID_EVENTBRICK == System.DBNull.Value) wtempstr = "ID_EVENTBRICK IS NULL";
					else wtempstr = "ID_EVENTBRICK = " + i_ID_EVENTBRICK.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_RECONSTRUCTION != null)
				{
					if (i_ID_RECONSTRUCTION == System.DBNull.Value) wtempstr = "ID_RECONSTRUCTION IS NULL";
					else wtempstr = "ID_RECONSTRUCTION = " + i_ID_RECONSTRUCTION.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_OPTION != null)
				{
					if (i_ID_OPTION == System.DBNull.Value) wtempstr = "ID_OPTION IS NULL";
					else wtempstr = "ID_OPTION = " + i_ID_OPTION.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID != null)
				{
					if (i_ID == System.DBNull.Value) wtempstr = "ID IS NULL";
					else wtempstr = "ID = " + i_ID.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_PROCESSOPERATION != null)
				{
					if (i_ID_PROCESSOPERATION == System.DBNull.Value) wtempstr = "ID_PROCESSOPERATION IS NULL";
					else wtempstr = "ID_PROCESSOPERATION = " + i_ID_PROCESSOPERATION.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_NAME != null)
				{
					if (i_NAME == System.DBNull.Value) wtempstr = "NAME IS NULL";
					else wtempstr = "NAME = " + i_NAME.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (order == OrderBy.Ascending) return SelectWhere(wherestr, "ID_EVENTBRICK ASC,ID_RECONSTRUCTION ASC,ID_OPTION ASC,ID ASC,ID_PROCESSOPERATION ASC,NAME ASC");
				else if (order == OrderBy.Descending) return SelectWhere(wherestr, "ID_EVENTBRICK DESC,ID_RECONSTRUCTION DESC,ID_OPTION DESC,ID DESC,ID_PROCESSOPERATION DESC,NAME DESC");
				return SelectWhere(wherestr, null);
			}
			/// <summary>
			/// Reads a set of rows from TB_VERTICES_ATTR and retrieves them into a new TB_VERTICES_ATTR object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the TB_VERTICES_ATTR class that can be used to read the retrieved data.</returns>
			static public TB_VERTICES_ATTR SelectWhere(string wherestr, string orderstr)
			{
				TB_VERTICES_ATTR newobj = new TB_VERTICES_ATTR();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_EVENTBRICK,ID_RECONSTRUCTION,ID_OPTION,ID,ID_PROCESSOPERATION,NAME,VALUE FROM TB_VERTICES_ATTR" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO TB_VERTICES_ATTR (ID_EVENTBRICK,ID_RECONSTRUCTION,ID_OPTION,ID,ID_PROCESSOPERATION,NAME,VALUE) VALUES (:p_1,:p_2,:p_3,:p_4,:p_5,:p_6,:p_7)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_EVENTBRICK;
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_RECONSTRUCTION;
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_OPTION;
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID;
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_PROCESSOPERATION;
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input).Value = a_NAME;
				newcmd.Parameters.Add("p_7", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_VALUE;
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the TB_VERTICES_FIT table in the DB.
		/// For data insertion, the Insert and Flush static methods are used. This class inserts data in batches.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class TB_VERTICES_FIT
		{
			internal TB_VERTICES_FIT() {}
			System.Data.DataRowCollection m_DRC;
			const int ArraySize = 100;
			static int index = 0;
			/// <summary>
			/// Since this class uses batch insertion, this method must be called to ensure that all rows are actually inserted into TB_VERTICES_FIT. Failure to do so will result in incomplete writes.
			/// </summary>
			static public void Flush() { if (cmd == null || index == 0) return; cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			private static long [] a_ID_EVENTBRICK = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_EVENTBRICK for the current row.
			/// </summary>
			public long _ID_EVENTBRICK
			{
				get
				{
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			private static long [] a_ID_RECONSTRUCTION = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_RECONSTRUCTION for the current row.
			/// </summary>
			public long _ID_RECONSTRUCTION
			{
				get
				{
					return System.Convert.ToInt64(m_DR[1]);
				}
			}
			private static long [] a_ID_OPTION = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_OPTION for the current row.
			/// </summary>
			public long _ID_OPTION
			{
				get
				{
					return System.Convert.ToInt64(m_DR[2]);
				}
			}
			private static long [] a_ID = new long[ArraySize];
			/// <summary>
			/// Retrieves ID for the current row.
			/// </summary>
			public long _ID
			{
				get
				{
					return System.Convert.ToInt64(m_DR[3]);
				}
			}
			private static string [] a_TYPE = new string[ArraySize];
			/// <summary>
			/// Retrieves TYPE for the current row.
			/// </summary>
			public string _TYPE
			{
				get
				{
					return System.Convert.ToString(m_DR[4]);
				}
			}
			private static double [] a_POSX = new double[ArraySize];
			/// <summary>
			/// Retrieves POSX for the current row.
			/// </summary>
			public double _POSX
			{
				get
				{
					return System.Convert.ToDouble(m_DR[5]);
				}
			}
			private static double [] a_POSY = new double[ArraySize];
			/// <summary>
			/// Retrieves POSY for the current row.
			/// </summary>
			public double _POSY
			{
				get
				{
					return System.Convert.ToDouble(m_DR[6]);
				}
			}
			private static double [] a_POSZ = new double[ArraySize];
			/// <summary>
			/// Retrieves POSZ for the current row.
			/// </summary>
			public double _POSZ
			{
				get
				{
					return System.Convert.ToDouble(m_DR[7]);
				}
			}
			private static double [] a_QUALITY = new double[ArraySize];
			/// <summary>
			/// Retrieves QUALITY for the current row.
			/// </summary>
			public double _QUALITY
			{
				get
				{
					return System.Convert.ToDouble(m_DR[8]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is actually inserted later, in a batch insertion command. The Flush method must be called to ensure all rows are actually written.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">the value to be inserted for ID_EVENTBRICK.</param>
			/// <param name="i_ID_RECONSTRUCTION">the value to be inserted for ID_RECONSTRUCTION.</param>
			/// <param name="i_ID_OPTION">the value to be inserted for ID_OPTION.</param>
			/// <param name="i_ID">the value to be inserted for ID.</param>
			/// <param name="i_TYPE">the value to be inserted for TYPE.</param>
			/// <param name="i_POSX">the value to be inserted for POSX.</param>
			/// <param name="i_POSY">the value to be inserted for POSY.</param>
			/// <param name="i_POSZ">the value to be inserted for POSZ.</param>
			/// <param name="i_QUALITY">the value to be inserted for QUALITY.</param>
			static public void Insert(long i_ID_EVENTBRICK,long i_ID_RECONSTRUCTION,long i_ID_OPTION,long i_ID,string i_TYPE,double i_POSX,double i_POSY,double i_POSZ,double i_QUALITY)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				a_ID_EVENTBRICK[index] = i_ID_EVENTBRICK;
				a_ID_RECONSTRUCTION[index] = i_ID_RECONSTRUCTION;
				a_ID_OPTION[index] = i_ID_OPTION;
				a_ID[index] = i_ID;
				a_TYPE[index] = i_TYPE;
				a_POSX[index] = i_POSX;
				a_POSY[index] = i_POSY;
				a_POSZ[index] = i_POSZ;
				a_QUALITY[index] = i_QUALITY;
				if (++index >= ArraySize) { cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			}
			/// <summary>
			/// Reads a set of rows from TB_VERTICES_FIT and retrieves them into a new TB_VERTICES_FIT object.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_RECONSTRUCTION">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_OPTION">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_TYPE">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="order">the ordering scheme to be applied to returned rows. This applies to all columns in the primary key.</param>
			/// <returns>a new instance of the TB_VERTICES_FIT class that can be used to read the retrieved data.</returns>
			static public TB_VERTICES_FIT SelectPrimaryKey(object i_ID_EVENTBRICK,object i_ID_RECONSTRUCTION,object i_ID_OPTION,object i_ID,object i_TYPE, OrderBy order)
			{
				string wherestr = "";
				string wtempstr = "";
				if (i_ID_EVENTBRICK != null)
				{
					if (i_ID_EVENTBRICK == System.DBNull.Value) wtempstr = "ID_EVENTBRICK IS NULL";
					else wtempstr = "ID_EVENTBRICK = " + i_ID_EVENTBRICK.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_RECONSTRUCTION != null)
				{
					if (i_ID_RECONSTRUCTION == System.DBNull.Value) wtempstr = "ID_RECONSTRUCTION IS NULL";
					else wtempstr = "ID_RECONSTRUCTION = " + i_ID_RECONSTRUCTION.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_OPTION != null)
				{
					if (i_ID_OPTION == System.DBNull.Value) wtempstr = "ID_OPTION IS NULL";
					else wtempstr = "ID_OPTION = " + i_ID_OPTION.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID != null)
				{
					if (i_ID == System.DBNull.Value) wtempstr = "ID IS NULL";
					else wtempstr = "ID = " + i_ID.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_TYPE != null)
				{
					if (i_TYPE == System.DBNull.Value) wtempstr = "TYPE IS NULL";
					else wtempstr = "TYPE = " + i_TYPE.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (order == OrderBy.Ascending) return SelectWhere(wherestr, "ID_EVENTBRICK ASC,ID_RECONSTRUCTION ASC,ID_OPTION ASC,ID ASC,TYPE ASC");
				else if (order == OrderBy.Descending) return SelectWhere(wherestr, "ID_EVENTBRICK DESC,ID_RECONSTRUCTION DESC,ID_OPTION DESC,ID DESC,TYPE DESC");
				return SelectWhere(wherestr, null);
			}
			/// <summary>
			/// Reads a set of rows from TB_VERTICES_FIT and retrieves them into a new TB_VERTICES_FIT object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the TB_VERTICES_FIT class that can be used to read the retrieved data.</returns>
			static public TB_VERTICES_FIT SelectWhere(string wherestr, string orderstr)
			{
				TB_VERTICES_FIT newobj = new TB_VERTICES_FIT();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_EVENTBRICK,ID_RECONSTRUCTION,ID_OPTION,ID,TYPE,POSX,POSY,POSZ,QUALITY FROM TB_VERTICES_FIT" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO TB_VERTICES_FIT (ID_EVENTBRICK,ID_RECONSTRUCTION,ID_OPTION,ID,TYPE,POSX,POSY,POSZ,QUALITY) VALUES (:p_1,:p_2,:p_3,:p_4,:p_5,:p_6,:p_7,:p_8,:p_9)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_EVENTBRICK;
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_RECONSTRUCTION;
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_OPTION;
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID;
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input).Value = a_TYPE;
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_POSX;
				newcmd.Parameters.Add("p_7", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_POSY;
				newcmd.Parameters.Add("p_8", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_POSZ;
				newcmd.Parameters.Add("p_9", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_QUALITY;
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the TB_VIEWS table in the DB.
		/// For data insertion, the Insert and Flush static methods are used. This class inserts data in batches.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class TB_VIEWS
		{
			internal TB_VIEWS() {}
			System.Data.DataRowCollection m_DRC;
			const int ArraySize = 100;
			static int index = 0;
			/// <summary>
			/// Since this class uses batch insertion, this method must be called to ensure that all rows are actually inserted into TB_VIEWS. Failure to do so will result in incomplete writes.
			/// </summary>
			static public void Flush() { if (cmd == null || index == 0) return; cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			private static long [] a_ID_EVENTBRICK = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_EVENTBRICK for the current row.
			/// </summary>
			public long _ID_EVENTBRICK
			{
				get
				{
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			private static long [] a_ID_ZONE = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_ZONE for the current row.
			/// </summary>
			public long _ID_ZONE
			{
				get
				{
					return System.Convert.ToInt64(m_DR[1]);
				}
			}
			private static int [] a_SIDE = new int[ArraySize];
			/// <summary>
			/// Retrieves SIDE for the current row.
			/// </summary>
			public int _SIDE
			{
				get
				{
					return System.Convert.ToInt32(m_DR[2]);
				}
			}
			private static long [] a_ID = new long[ArraySize];
			/// <summary>
			/// Retrieves ID for the current row.
			/// </summary>
			public long _ID
			{
				get
				{
					return System.Convert.ToInt64(m_DR[3]);
				}
			}
			private static double [] a_DOWNZ = new double[ArraySize];
			/// <summary>
			/// Retrieves DOWNZ for the current row.
			/// </summary>
			public double _DOWNZ
			{
				get
				{
					return System.Convert.ToDouble(m_DR[4]);
				}
			}
			private static double [] a_UPZ = new double[ArraySize];
			/// <summary>
			/// Retrieves UPZ for the current row.
			/// </summary>
			public double _UPZ
			{
				get
				{
					return System.Convert.ToDouble(m_DR[5]);
				}
			}
			private static double [] a_POSX = new double[ArraySize];
			/// <summary>
			/// Retrieves POSX for the current row.
			/// </summary>
			public double _POSX
			{
				get
				{
					return System.Convert.ToDouble(m_DR[6]);
				}
			}
			private static double [] a_POSY = new double[ArraySize];
			/// <summary>
			/// Retrieves POSY for the current row.
			/// </summary>
			public double _POSY
			{
				get
				{
					return System.Convert.ToDouble(m_DR[7]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is actually inserted later, in a batch insertion command. The Flush method must be called to ensure all rows are actually written.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">the value to be inserted for ID_EVENTBRICK.</param>
			/// <param name="i_ID_ZONE">the value to be inserted for ID_ZONE.</param>
			/// <param name="i_SIDE">the value to be inserted for SIDE.</param>
			/// <param name="i_ID">the value to be inserted for ID.</param>
			/// <param name="i_DOWNZ">the value to be inserted for DOWNZ.</param>
			/// <param name="i_UPZ">the value to be inserted for UPZ.</param>
			/// <param name="i_POSX">the value to be inserted for POSX.</param>
			/// <param name="i_POSY">the value to be inserted for POSY.</param>
			static public void Insert(long i_ID_EVENTBRICK,long i_ID_ZONE,int i_SIDE,long i_ID,double i_DOWNZ,double i_UPZ,double i_POSX,double i_POSY)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				a_ID_EVENTBRICK[index] = i_ID_EVENTBRICK;
				a_ID_ZONE[index] = i_ID_ZONE;
				a_SIDE[index] = i_SIDE;
				a_ID[index] = i_ID;
				a_DOWNZ[index] = i_DOWNZ;
				a_UPZ[index] = i_UPZ;
				a_POSX[index] = i_POSX;
				a_POSY[index] = i_POSY;
				if (++index >= ArraySize) { cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			}
			/// <summary>
			/// Reads a set of rows from TB_VIEWS and retrieves them into a new TB_VIEWS object.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_ZONE">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_SIDE">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="order">the ordering scheme to be applied to returned rows. This applies to all columns in the primary key.</param>
			/// <returns>a new instance of the TB_VIEWS class that can be used to read the retrieved data.</returns>
			static public TB_VIEWS SelectPrimaryKey(object i_ID_EVENTBRICK,object i_ID_ZONE,object i_SIDE,object i_ID, OrderBy order)
			{
				string wherestr = "";
				string wtempstr = "";
				if (i_ID_EVENTBRICK != null)
				{
					if (i_ID_EVENTBRICK == System.DBNull.Value) wtempstr = "ID_EVENTBRICK IS NULL";
					else wtempstr = "ID_EVENTBRICK = " + i_ID_EVENTBRICK.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_ZONE != null)
				{
					if (i_ID_ZONE == System.DBNull.Value) wtempstr = "ID_ZONE IS NULL";
					else wtempstr = "ID_ZONE = " + i_ID_ZONE.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_SIDE != null)
				{
					if (i_SIDE == System.DBNull.Value) wtempstr = "SIDE IS NULL";
					else wtempstr = "SIDE = " + i_SIDE.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID != null)
				{
					if (i_ID == System.DBNull.Value) wtempstr = "ID IS NULL";
					else wtempstr = "ID = " + i_ID.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (order == OrderBy.Ascending) return SelectWhere(wherestr, "ID_EVENTBRICK ASC,ID_ZONE ASC,SIDE ASC,ID ASC");
				else if (order == OrderBy.Descending) return SelectWhere(wherestr, "ID_EVENTBRICK DESC,ID_ZONE DESC,SIDE DESC,ID DESC");
				return SelectWhere(wherestr, null);
			}
			/// <summary>
			/// Reads a set of rows from TB_VIEWS and retrieves them into a new TB_VIEWS object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the TB_VIEWS class that can be used to read the retrieved data.</returns>
			static public TB_VIEWS SelectWhere(string wherestr, string orderstr)
			{
				TB_VIEWS newobj = new TB_VIEWS();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_EVENTBRICK,ID_ZONE,SIDE,ID,DOWNZ,UPZ,POSX,POSY FROM TB_VIEWS" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO TB_VIEWS (ID_EVENTBRICK,ID_ZONE,SIDE,ID,DOWNZ,UPZ,POSX,POSY) VALUES (:p_1,:p_2,:p_3,:p_4,:p_5,:p_6,:p_7,:p_8)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_EVENTBRICK;
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_ZONE;
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Int, System.Data.ParameterDirection.Input).Value = a_SIDE;
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID;
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_DOWNZ;
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_UPZ;
				newcmd.Parameters.Add("p_7", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_POSX;
				newcmd.Parameters.Add("p_8", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_POSY;
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the TB_VOLTKS_ALIGNMUTKS table in the DB.
		/// For data insertion, the Insert and Flush static methods are used. This class inserts data in batches.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class TB_VOLTKS_ALIGNMUTKS
		{
			internal TB_VOLTKS_ALIGNMUTKS() {}
			System.Data.DataRowCollection m_DRC;
			const int ArraySize = 100;
			static int index = 0;
			/// <summary>
			/// Since this class uses batch insertion, this method must be called to ensure that all rows are actually inserted into TB_VOLTKS_ALIGNMUTKS. Failure to do so will result in incomplete writes.
			/// </summary>
			static public void Flush() { if (cmd == null || index == 0) return; cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			private static long [] a_ID_EVENTBRICK = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_EVENTBRICK for the current row.
			/// </summary>
			public long _ID_EVENTBRICK
			{
				get
				{
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			private static long [] a_ID_RECONSTRUCTION = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_RECONSTRUCTION for the current row.
			/// </summary>
			public long _ID_RECONSTRUCTION
			{
				get
				{
					return System.Convert.ToInt64(m_DR[1]);
				}
			}
			private static long [] a_ID_OPTION = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_OPTION for the current row.
			/// </summary>
			public long _ID_OPTION
			{
				get
				{
					return System.Convert.ToInt64(m_DR[2]);
				}
			}
			private static long [] a_ID_PLATE = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_PLATE for the current row.
			/// </summary>
			public long _ID_PLATE
			{
				get
				{
					return System.Convert.ToInt64(m_DR[3]);
				}
			}
			private static long [] a_ID_ZONE = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_ZONE for the current row.
			/// </summary>
			public long _ID_ZONE
			{
				get
				{
					return System.Convert.ToInt64(m_DR[4]);
				}
			}
			private static int [] a_SIDE = new int[ArraySize];
			/// <summary>
			/// Retrieves SIDE for the current row.
			/// </summary>
			public int _SIDE
			{
				get
				{
					return System.Convert.ToInt32(m_DR[5]);
				}
			}
			private static long [] a_ID_MIPMICROTRACK = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_MIPMICROTRACK for the current row.
			/// </summary>
			public long _ID_MIPMICROTRACK
			{
				get
				{
					return System.Convert.ToInt64(m_DR[6]);
				}
			}
			private static long [] a_ID_VOLUMETRACK = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_VOLUMETRACK for the current row.
			/// </summary>
			public long _ID_VOLUMETRACK
			{
				get
				{
					return System.Convert.ToInt64(m_DR[7]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is actually inserted later, in a batch insertion command. The Flush method must be called to ensure all rows are actually written.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">the value to be inserted for ID_EVENTBRICK.</param>
			/// <param name="i_ID_RECONSTRUCTION">the value to be inserted for ID_RECONSTRUCTION.</param>
			/// <param name="i_ID_OPTION">the value to be inserted for ID_OPTION.</param>
			/// <param name="i_ID_PLATE">the value to be inserted for ID_PLATE.</param>
			/// <param name="i_ID_ZONE">the value to be inserted for ID_ZONE.</param>
			/// <param name="i_SIDE">the value to be inserted for SIDE.</param>
			/// <param name="i_ID_MIPMICROTRACK">the value to be inserted for ID_MIPMICROTRACK.</param>
			/// <param name="i_ID_VOLUMETRACK">the value to be inserted for ID_VOLUMETRACK.</param>
			static public void Insert(long i_ID_EVENTBRICK,long i_ID_RECONSTRUCTION,long i_ID_OPTION,long i_ID_PLATE,long i_ID_ZONE,int i_SIDE,long i_ID_MIPMICROTRACK,long i_ID_VOLUMETRACK)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				a_ID_EVENTBRICK[index] = i_ID_EVENTBRICK;
				a_ID_RECONSTRUCTION[index] = i_ID_RECONSTRUCTION;
				a_ID_OPTION[index] = i_ID_OPTION;
				a_ID_PLATE[index] = i_ID_PLATE;
				a_ID_ZONE[index] = i_ID_ZONE;
				a_SIDE[index] = i_SIDE;
				a_ID_MIPMICROTRACK[index] = i_ID_MIPMICROTRACK;
				a_ID_VOLUMETRACK[index] = i_ID_VOLUMETRACK;
				if (++index >= ArraySize) { cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			}
			/// <summary>
			/// Reads a set of rows from TB_VOLTKS_ALIGNMUTKS and retrieves them into a new TB_VOLTKS_ALIGNMUTKS object.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_RECONSTRUCTION">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_OPTION">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_PLATE">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_SIDE">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_VOLUMETRACK">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="order">the ordering scheme to be applied to returned rows. This applies to all columns in the primary key.</param>
			/// <returns>a new instance of the TB_VOLTKS_ALIGNMUTKS class that can be used to read the retrieved data.</returns>
			static public TB_VOLTKS_ALIGNMUTKS SelectPrimaryKey(object i_ID_EVENTBRICK,object i_ID_RECONSTRUCTION,object i_ID_OPTION,object i_ID_PLATE,object i_SIDE,object i_ID_VOLUMETRACK, OrderBy order)
			{
				string wherestr = "";
				string wtempstr = "";
				if (i_ID_EVENTBRICK != null)
				{
					if (i_ID_EVENTBRICK == System.DBNull.Value) wtempstr = "ID_EVENTBRICK IS NULL";
					else wtempstr = "ID_EVENTBRICK = " + i_ID_EVENTBRICK.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_RECONSTRUCTION != null)
				{
					if (i_ID_RECONSTRUCTION == System.DBNull.Value) wtempstr = "ID_RECONSTRUCTION IS NULL";
					else wtempstr = "ID_RECONSTRUCTION = " + i_ID_RECONSTRUCTION.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_OPTION != null)
				{
					if (i_ID_OPTION == System.DBNull.Value) wtempstr = "ID_OPTION IS NULL";
					else wtempstr = "ID_OPTION = " + i_ID_OPTION.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_PLATE != null)
				{
					if (i_ID_PLATE == System.DBNull.Value) wtempstr = "ID_PLATE IS NULL";
					else wtempstr = "ID_PLATE = " + i_ID_PLATE.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_SIDE != null)
				{
					if (i_SIDE == System.DBNull.Value) wtempstr = "SIDE IS NULL";
					else wtempstr = "SIDE = " + i_SIDE.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_VOLUMETRACK != null)
				{
					if (i_ID_VOLUMETRACK == System.DBNull.Value) wtempstr = "ID_VOLUMETRACK IS NULL";
					else wtempstr = "ID_VOLUMETRACK = " + i_ID_VOLUMETRACK.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (order == OrderBy.Ascending) return SelectWhere(wherestr, "ID_EVENTBRICK ASC,ID_RECONSTRUCTION ASC,ID_OPTION ASC,ID_PLATE ASC,SIDE ASC,ID_VOLUMETRACK ASC");
				else if (order == OrderBy.Descending) return SelectWhere(wherestr, "ID_EVENTBRICK DESC,ID_RECONSTRUCTION DESC,ID_OPTION DESC,ID_PLATE DESC,SIDE DESC,ID_VOLUMETRACK DESC");
				return SelectWhere(wherestr, null);
			}
			/// <summary>
			/// Reads a set of rows from TB_VOLTKS_ALIGNMUTKS and retrieves them into a new TB_VOLTKS_ALIGNMUTKS object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the TB_VOLTKS_ALIGNMUTKS class that can be used to read the retrieved data.</returns>
			static public TB_VOLTKS_ALIGNMUTKS SelectWhere(string wherestr, string orderstr)
			{
				TB_VOLTKS_ALIGNMUTKS newobj = new TB_VOLTKS_ALIGNMUTKS();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_EVENTBRICK,ID_RECONSTRUCTION,ID_OPTION,ID_PLATE,ID_ZONE,SIDE,ID_MIPMICROTRACK,ID_VOLUMETRACK FROM TB_VOLTKS_ALIGNMUTKS" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO TB_VOLTKS_ALIGNMUTKS (ID_EVENTBRICK,ID_RECONSTRUCTION,ID_OPTION,ID_PLATE,ID_ZONE,SIDE,ID_MIPMICROTRACK,ID_VOLUMETRACK) VALUES (:p_1,:p_2,:p_3,:p_4,:p_5,:p_6,:p_7,:p_8)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_EVENTBRICK;
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_RECONSTRUCTION;
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_OPTION;
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_PLATE;
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_ZONE;
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.Int, System.Data.ParameterDirection.Input).Value = a_SIDE;
				newcmd.Parameters.Add("p_7", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_MIPMICROTRACK;
				newcmd.Parameters.Add("p_8", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_VOLUMETRACK;
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the TB_VOLUMES table in the DB.
		/// For data insertion, the Insert method is used. Rows are inserted one by one.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class TB_VOLUMES
		{
			internal TB_VOLUMES() {}
			System.Data.DataRowCollection m_DRC;
			/// <summary>
			/// Retrieves ID_EVENTBRICK for the current row.
			/// </summary>
			public long _ID_EVENTBRICK
			{
				get
				{
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			/// <summary>
			/// Retrieves VOLUME for the current row.
			/// </summary>
			public long _VOLUME
			{
				get
				{
					return System.Convert.ToInt64(m_DR[1]);
				}
			}
			/// <summary>
			/// Retrieves ID_PROCESSOPERATION for the current row.
			/// </summary>
			public long _ID_PROCESSOPERATION
			{
				get
				{
					return System.Convert.ToInt64(m_DR[2]);
				}
			}
			/// <summary>
			/// Retrieves ID for the current row.
			/// </summary>
			public long _ID
			{
				get
				{
					return System.Convert.ToInt64(m_DR[3]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is inserted immediately.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">the value to be inserted for ID_EVENTBRICK.</param>
			/// <param name="i_VOLUME">the value to be inserted for VOLUME.</param>
			/// <param name="i_ID_PROCESSOPERATION">the value to be inserted for ID_PROCESSOPERATION.</param>
			/// <param name="i_ID">the value to be inserted for ID. This value is actually used only if this method call is involved in data publication/replication, otherwise the actual value is generated by the OPERA DB and the supplied value is ignored.</param>
			/// <returns>the value of ID for the new row.</returns>
			static public long Insert(long i_ID_EVENTBRICK,long i_VOLUME,long i_ID_PROCESSOPERATION,long i_ID)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = i_ID_EVENTBRICK;
				cmd.Parameters[1].Value = i_VOLUME;
				cmd.Parameters[2].Value = i_ID_PROCESSOPERATION;
				cmd.Parameters[3].Value = i_ID;
				cmd.ExecuteNonQuery();
				return SySal.OperaDb.Convert.ToInt64(cmd.Parameters[4].Value);
			}
			/// <summary>
			/// Reads a set of rows from TB_VOLUMES and retrieves them into a new TB_VOLUMES object.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_VOLUME">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_PROCESSOPERATION">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="order">the ordering scheme to be applied to returned rows. This applies to all columns in the primary key.</param>
			/// <returns>a new instance of the TB_VOLUMES class that can be used to read the retrieved data.</returns>
			static public TB_VOLUMES SelectPrimaryKey(object i_ID_EVENTBRICK,object i_VOLUME,object i_ID_PROCESSOPERATION, OrderBy order)
			{
				string wherestr = "";
				string wtempstr = "";
				if (i_ID_EVENTBRICK != null)
				{
					if (i_ID_EVENTBRICK == System.DBNull.Value) wtempstr = "ID_EVENTBRICK IS NULL";
					else wtempstr = "ID_EVENTBRICK = " + i_ID_EVENTBRICK.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_VOLUME != null)
				{
					if (i_VOLUME == System.DBNull.Value) wtempstr = "VOLUME IS NULL";
					else wtempstr = "VOLUME = " + i_VOLUME.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_PROCESSOPERATION != null)
				{
					if (i_ID_PROCESSOPERATION == System.DBNull.Value) wtempstr = "ID_PROCESSOPERATION IS NULL";
					else wtempstr = "ID_PROCESSOPERATION = " + i_ID_PROCESSOPERATION.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (order == OrderBy.Ascending) return SelectWhere(wherestr, "ID_EVENTBRICK ASC,VOLUME ASC,ID_PROCESSOPERATION ASC");
				else if (order == OrderBy.Descending) return SelectWhere(wherestr, "ID_EVENTBRICK DESC,VOLUME DESC,ID_PROCESSOPERATION DESC");
				return SelectWhere(wherestr, null);
			}
			/// <summary>
			/// Reads a set of rows from TB_VOLUMES and retrieves them into a new TB_VOLUMES object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the TB_VOLUMES class that can be used to read the retrieved data.</returns>
			static public TB_VOLUMES SelectWhere(string wherestr, string orderstr)
			{
				TB_VOLUMES newobj = new TB_VOLUMES();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_EVENTBRICK,VOLUME,ID_PROCESSOPERATION,ID FROM TB_VOLUMES" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO TB_VOLUMES (ID_EVENTBRICK,VOLUME,ID_PROCESSOPERATION,ID) VALUES (:p_1,:p_2,:p_3,:p_4) RETURNING ID INTO :o_4");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("o_4", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Output);
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the TB_VOLUMETRACKS table in the DB.
		/// For data insertion, the Insert and Flush static methods are used. This class inserts data in batches.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class TB_VOLUMETRACKS
		{
			internal TB_VOLUMETRACKS() {}
			System.Data.DataRowCollection m_DRC;
			const int ArraySize = 100;
			static int index = 0;
			/// <summary>
			/// Since this class uses batch insertion, this method must be called to ensure that all rows are actually inserted into TB_VOLUMETRACKS. Failure to do so will result in incomplete writes.
			/// </summary>
			static public void Flush() { if (cmd == null || index == 0) return; cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			private static long [] a_ID_EVENTBRICK = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_EVENTBRICK for the current row.
			/// </summary>
			public long _ID_EVENTBRICK
			{
				get
				{
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			private static long [] a_ID_RECONSTRUCTION = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_RECONSTRUCTION for the current row.
			/// </summary>
			public long _ID_RECONSTRUCTION
			{
				get
				{
					return System.Convert.ToInt64(m_DR[1]);
				}
			}
			private static long [] a_ID_OPTION = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_OPTION for the current row.
			/// </summary>
			public long _ID_OPTION
			{
				get
				{
					return System.Convert.ToInt64(m_DR[2]);
				}
			}
			private static long [] a_ID = new long[ArraySize];
			/// <summary>
			/// Retrieves ID for the current row.
			/// </summary>
			public long _ID
			{
				get
				{
					return System.Convert.ToInt64(m_DR[3]);
				}
			}
			private static double [] a_DOWNZ = new double[ArraySize];
			/// <summary>
			/// Retrieves DOWNZ for the current row.
			/// </summary>
			public double _DOWNZ
			{
				get
				{
					return System.Convert.ToDouble(m_DR[4]);
				}
			}
			private static double [] a_UPZ = new double[ArraySize];
			/// <summary>
			/// Retrieves UPZ for the current row.
			/// </summary>
			public double _UPZ
			{
				get
				{
					return System.Convert.ToDouble(m_DR[5]);
				}
			}
			private static object [] a_ID_DOWNSTREAMVERTEX = new object[ArraySize];
			/// <summary>
			/// Retrieves ID_DOWNSTREAMVERTEX for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _ID_DOWNSTREAMVERTEX
			{
				get
				{
					if (m_DR[6] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[6]);
				}
			}
			private static object [] a_ID_UPSTREAMVERTEX = new object[ArraySize];
			/// <summary>
			/// Retrieves ID_UPSTREAMVERTEX for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _ID_UPSTREAMVERTEX
			{
				get
				{
					if (m_DR[7] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[7]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is actually inserted later, in a batch insertion command. The Flush method must be called to ensure all rows are actually written.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">the value to be inserted for ID_EVENTBRICK.</param>
			/// <param name="i_ID_RECONSTRUCTION">the value to be inserted for ID_RECONSTRUCTION.</param>
			/// <param name="i_ID_OPTION">the value to be inserted for ID_OPTION.</param>
			/// <param name="i_ID">the value to be inserted for ID.</param>
			/// <param name="i_DOWNZ">the value to be inserted for DOWNZ.</param>
			/// <param name="i_UPZ">the value to be inserted for UPZ.</param>
			/// <param name="i_ID_DOWNSTREAMVERTEX">the value to be inserted for ID_DOWNSTREAMVERTEX. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_ID_UPSTREAMVERTEX">the value to be inserted for ID_UPSTREAMVERTEX. The value for this parameter can be long or System.DBNull.Value.</param>
			static public void Insert(long i_ID_EVENTBRICK,long i_ID_RECONSTRUCTION,long i_ID_OPTION,long i_ID,double i_DOWNZ,double i_UPZ,object i_ID_DOWNSTREAMVERTEX,object i_ID_UPSTREAMVERTEX)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				a_ID_EVENTBRICK[index] = i_ID_EVENTBRICK;
				a_ID_RECONSTRUCTION[index] = i_ID_RECONSTRUCTION;
				a_ID_OPTION[index] = i_ID_OPTION;
				a_ID[index] = i_ID;
				a_DOWNZ[index] = i_DOWNZ;
				a_UPZ[index] = i_UPZ;
				a_ID_DOWNSTREAMVERTEX[index] = i_ID_DOWNSTREAMVERTEX;
				a_ID_UPSTREAMVERTEX[index] = i_ID_UPSTREAMVERTEX;
				if (++index >= ArraySize) { cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			}
			/// <summary>
			/// Reads a set of rows from TB_VOLUMETRACKS and retrieves them into a new TB_VOLUMETRACKS object.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_RECONSTRUCTION">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_OPTION">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="order">the ordering scheme to be applied to returned rows. This applies to all columns in the primary key.</param>
			/// <returns>a new instance of the TB_VOLUMETRACKS class that can be used to read the retrieved data.</returns>
			static public TB_VOLUMETRACKS SelectPrimaryKey(object i_ID_EVENTBRICK,object i_ID_RECONSTRUCTION,object i_ID_OPTION,object i_ID, OrderBy order)
			{
				string wherestr = "";
				string wtempstr = "";
				if (i_ID_EVENTBRICK != null)
				{
					if (i_ID_EVENTBRICK == System.DBNull.Value) wtempstr = "ID_EVENTBRICK IS NULL";
					else wtempstr = "ID_EVENTBRICK = " + i_ID_EVENTBRICK.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_RECONSTRUCTION != null)
				{
					if (i_ID_RECONSTRUCTION == System.DBNull.Value) wtempstr = "ID_RECONSTRUCTION IS NULL";
					else wtempstr = "ID_RECONSTRUCTION = " + i_ID_RECONSTRUCTION.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_OPTION != null)
				{
					if (i_ID_OPTION == System.DBNull.Value) wtempstr = "ID_OPTION IS NULL";
					else wtempstr = "ID_OPTION = " + i_ID_OPTION.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID != null)
				{
					if (i_ID == System.DBNull.Value) wtempstr = "ID IS NULL";
					else wtempstr = "ID = " + i_ID.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (order == OrderBy.Ascending) return SelectWhere(wherestr, "ID_EVENTBRICK ASC,ID_RECONSTRUCTION ASC,ID_OPTION ASC,ID ASC");
				else if (order == OrderBy.Descending) return SelectWhere(wherestr, "ID_EVENTBRICK DESC,ID_RECONSTRUCTION DESC,ID_OPTION DESC,ID DESC");
				return SelectWhere(wherestr, null);
			}
			/// <summary>
			/// Reads a set of rows from TB_VOLUMETRACKS and retrieves them into a new TB_VOLUMETRACKS object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the TB_VOLUMETRACKS class that can be used to read the retrieved data.</returns>
			static public TB_VOLUMETRACKS SelectWhere(string wherestr, string orderstr)
			{
				TB_VOLUMETRACKS newobj = new TB_VOLUMETRACKS();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_EVENTBRICK,ID_RECONSTRUCTION,ID_OPTION,ID,DOWNZ,UPZ,ID_DOWNSTREAMVERTEX,ID_UPSTREAMVERTEX FROM TB_VOLUMETRACKS" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO TB_VOLUMETRACKS (ID_EVENTBRICK,ID_RECONSTRUCTION,ID_OPTION,ID,DOWNZ,UPZ,ID_DOWNSTREAMVERTEX,ID_UPSTREAMVERTEX) VALUES (:p_1,:p_2,:p_3,:p_4,:p_5,:p_6,:p_7,:p_8)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_EVENTBRICK;
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_RECONSTRUCTION;
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_OPTION;
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID;
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_DOWNZ;
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_UPZ;
				newcmd.Parameters.Add("p_7", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_DOWNSTREAMVERTEX;
				newcmd.Parameters.Add("p_8", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_UPSTREAMVERTEX;
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the TB_VOLUMETRACKS_ATTR table in the DB.
		/// For data insertion, the Insert and Flush static methods are used. This class inserts data in batches.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class TB_VOLUMETRACKS_ATTR
		{
			internal TB_VOLUMETRACKS_ATTR() {}
			System.Data.DataRowCollection m_DRC;
			const int ArraySize = 100;
			static int index = 0;
			/// <summary>
			/// Since this class uses batch insertion, this method must be called to ensure that all rows are actually inserted into TB_VOLUMETRACKS_ATTR. Failure to do so will result in incomplete writes.
			/// </summary>
			static public void Flush() { if (cmd == null || index == 0) return; cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			private static long [] a_ID_EVENTBRICK = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_EVENTBRICK for the current row.
			/// </summary>
			public long _ID_EVENTBRICK
			{
				get
				{
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			private static long [] a_ID_RECONSTRUCTION = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_RECONSTRUCTION for the current row.
			/// </summary>
			public long _ID_RECONSTRUCTION
			{
				get
				{
					return System.Convert.ToInt64(m_DR[1]);
				}
			}
			private static long [] a_ID_OPTION = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_OPTION for the current row.
			/// </summary>
			public long _ID_OPTION
			{
				get
				{
					return System.Convert.ToInt64(m_DR[2]);
				}
			}
			private static long [] a_ID = new long[ArraySize];
			/// <summary>
			/// Retrieves ID for the current row.
			/// </summary>
			public long _ID
			{
				get
				{
					return System.Convert.ToInt64(m_DR[3]);
				}
			}
			private static long [] a_ID_PROCESSOPERATION = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_PROCESSOPERATION for the current row.
			/// </summary>
			public long _ID_PROCESSOPERATION
			{
				get
				{
					return System.Convert.ToInt64(m_DR[4]);
				}
			}
			private static string [] a_NAME = new string[ArraySize];
			/// <summary>
			/// Retrieves NAME for the current row.
			/// </summary>
			public string _NAME
			{
				get
				{
					return System.Convert.ToString(m_DR[5]);
				}
			}
			private static object [] a_VALUE = new object[ArraySize];
			/// <summary>
			/// Retrieves VALUE for the current row. The return value can be System.DBNull.Value or a value that can be cast to double.
			/// </summary>
			public object _VALUE
			{
				get
				{
					if (m_DR[6] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDouble(m_DR[6]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is actually inserted later, in a batch insertion command. The Flush method must be called to ensure all rows are actually written.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">the value to be inserted for ID_EVENTBRICK.</param>
			/// <param name="i_ID_RECONSTRUCTION">the value to be inserted for ID_RECONSTRUCTION.</param>
			/// <param name="i_ID_OPTION">the value to be inserted for ID_OPTION.</param>
			/// <param name="i_ID">the value to be inserted for ID.</param>
			/// <param name="i_ID_PROCESSOPERATION">the value to be inserted for ID_PROCESSOPERATION.</param>
			/// <param name="i_NAME">the value to be inserted for NAME.</param>
			/// <param name="i_VALUE">the value to be inserted for VALUE. The value for this parameter can be double or System.DBNull.Value.</param>
			static public void Insert(long i_ID_EVENTBRICK,long i_ID_RECONSTRUCTION,long i_ID_OPTION,long i_ID,long i_ID_PROCESSOPERATION,string i_NAME,object i_VALUE)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				a_ID_EVENTBRICK[index] = i_ID_EVENTBRICK;
				a_ID_RECONSTRUCTION[index] = i_ID_RECONSTRUCTION;
				a_ID_OPTION[index] = i_ID_OPTION;
				a_ID[index] = i_ID;
				a_ID_PROCESSOPERATION[index] = i_ID_PROCESSOPERATION;
				a_NAME[index] = i_NAME;
				a_VALUE[index] = i_VALUE;
				if (++index >= ArraySize) { cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			}
			/// <summary>
			/// Reads a set of rows from TB_VOLUMETRACKS_ATTR and retrieves them into a new TB_VOLUMETRACKS_ATTR object.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_RECONSTRUCTION">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_OPTION">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_PROCESSOPERATION">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_NAME">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="order">the ordering scheme to be applied to returned rows. This applies to all columns in the primary key.</param>
			/// <returns>a new instance of the TB_VOLUMETRACKS_ATTR class that can be used to read the retrieved data.</returns>
			static public TB_VOLUMETRACKS_ATTR SelectPrimaryKey(object i_ID_EVENTBRICK,object i_ID_RECONSTRUCTION,object i_ID_OPTION,object i_ID,object i_ID_PROCESSOPERATION,object i_NAME, OrderBy order)
			{
				string wherestr = "";
				string wtempstr = "";
				if (i_ID_EVENTBRICK != null)
				{
					if (i_ID_EVENTBRICK == System.DBNull.Value) wtempstr = "ID_EVENTBRICK IS NULL";
					else wtempstr = "ID_EVENTBRICK = " + i_ID_EVENTBRICK.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_RECONSTRUCTION != null)
				{
					if (i_ID_RECONSTRUCTION == System.DBNull.Value) wtempstr = "ID_RECONSTRUCTION IS NULL";
					else wtempstr = "ID_RECONSTRUCTION = " + i_ID_RECONSTRUCTION.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_OPTION != null)
				{
					if (i_ID_OPTION == System.DBNull.Value) wtempstr = "ID_OPTION IS NULL";
					else wtempstr = "ID_OPTION = " + i_ID_OPTION.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID != null)
				{
					if (i_ID == System.DBNull.Value) wtempstr = "ID IS NULL";
					else wtempstr = "ID = " + i_ID.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_PROCESSOPERATION != null)
				{
					if (i_ID_PROCESSOPERATION == System.DBNull.Value) wtempstr = "ID_PROCESSOPERATION IS NULL";
					else wtempstr = "ID_PROCESSOPERATION = " + i_ID_PROCESSOPERATION.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_NAME != null)
				{
					if (i_NAME == System.DBNull.Value) wtempstr = "NAME IS NULL";
					else wtempstr = "NAME = " + i_NAME.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (order == OrderBy.Ascending) return SelectWhere(wherestr, "ID_EVENTBRICK ASC,ID_RECONSTRUCTION ASC,ID_OPTION ASC,ID ASC,ID_PROCESSOPERATION ASC,NAME ASC");
				else if (order == OrderBy.Descending) return SelectWhere(wherestr, "ID_EVENTBRICK DESC,ID_RECONSTRUCTION DESC,ID_OPTION DESC,ID DESC,ID_PROCESSOPERATION DESC,NAME DESC");
				return SelectWhere(wherestr, null);
			}
			/// <summary>
			/// Reads a set of rows from TB_VOLUMETRACKS_ATTR and retrieves them into a new TB_VOLUMETRACKS_ATTR object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the TB_VOLUMETRACKS_ATTR class that can be used to read the retrieved data.</returns>
			static public TB_VOLUMETRACKS_ATTR SelectWhere(string wherestr, string orderstr)
			{
				TB_VOLUMETRACKS_ATTR newobj = new TB_VOLUMETRACKS_ATTR();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_EVENTBRICK,ID_RECONSTRUCTION,ID_OPTION,ID,ID_PROCESSOPERATION,NAME,VALUE FROM TB_VOLUMETRACKS_ATTR" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO TB_VOLUMETRACKS_ATTR (ID_EVENTBRICK,ID_RECONSTRUCTION,ID_OPTION,ID,ID_PROCESSOPERATION,NAME,VALUE) VALUES (:p_1,:p_2,:p_3,:p_4,:p_5,:p_6,:p_7)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_EVENTBRICK;
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_RECONSTRUCTION;
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_OPTION;
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID;
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_PROCESSOPERATION;
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input).Value = a_NAME;
				newcmd.Parameters.Add("p_7", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_VALUE;
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the TB_VOLUMETRACKS_FIT table in the DB.
		/// For data insertion, the Insert and Flush static methods are used. This class inserts data in batches.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class TB_VOLUMETRACKS_FIT
		{
			internal TB_VOLUMETRACKS_FIT() {}
			System.Data.DataRowCollection m_DRC;
			const int ArraySize = 100;
			static int index = 0;
			/// <summary>
			/// Since this class uses batch insertion, this method must be called to ensure that all rows are actually inserted into TB_VOLUMETRACKS_FIT. Failure to do so will result in incomplete writes.
			/// </summary>
			static public void Flush() { if (cmd == null || index == 0) return; cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			private static long [] a_ID_EVENTBRICK = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_EVENTBRICK for the current row.
			/// </summary>
			public long _ID_EVENTBRICK
			{
				get
				{
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			private static long [] a_ID_RECONSTRUCTION = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_RECONSTRUCTION for the current row.
			/// </summary>
			public long _ID_RECONSTRUCTION
			{
				get
				{
					return System.Convert.ToInt64(m_DR[1]);
				}
			}
			private static long [] a_ID_OPTION = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_OPTION for the current row.
			/// </summary>
			public long _ID_OPTION
			{
				get
				{
					return System.Convert.ToInt64(m_DR[2]);
				}
			}
			private static long [] a_ID = new long[ArraySize];
			/// <summary>
			/// Retrieves ID for the current row.
			/// </summary>
			public long _ID
			{
				get
				{
					return System.Convert.ToInt64(m_DR[3]);
				}
			}
			private static string [] a_TYPE = new string[ArraySize];
			/// <summary>
			/// Retrieves TYPE for the current row.
			/// </summary>
			public string _TYPE
			{
				get
				{
					return System.Convert.ToString(m_DR[4]);
				}
			}
			private static double [] a_DOWNPOSX = new double[ArraySize];
			/// <summary>
			/// Retrieves DOWNPOSX for the current row.
			/// </summary>
			public double _DOWNPOSX
			{
				get
				{
					return System.Convert.ToDouble(m_DR[5]);
				}
			}
			private static double [] a_DOWNPOSY = new double[ArraySize];
			/// <summary>
			/// Retrieves DOWNPOSY for the current row.
			/// </summary>
			public double _DOWNPOSY
			{
				get
				{
					return System.Convert.ToDouble(m_DR[6]);
				}
			}
			private static double [] a_DOWNPOSZ = new double[ArraySize];
			/// <summary>
			/// Retrieves DOWNPOSZ for the current row.
			/// </summary>
			public double _DOWNPOSZ
			{
				get
				{
					return System.Convert.ToDouble(m_DR[7]);
				}
			}
			private static double [] a_DOWNSLOPEX = new double[ArraySize];
			/// <summary>
			/// Retrieves DOWNSLOPEX for the current row.
			/// </summary>
			public double _DOWNSLOPEX
			{
				get
				{
					return System.Convert.ToDouble(m_DR[8]);
				}
			}
			private static double [] a_DOWNSLOPEY = new double[ArraySize];
			/// <summary>
			/// Retrieves DOWNSLOPEY for the current row.
			/// </summary>
			public double _DOWNSLOPEY
			{
				get
				{
					return System.Convert.ToDouble(m_DR[9]);
				}
			}
			private static double [] a_DOWNQUALITY = new double[ArraySize];
			/// <summary>
			/// Retrieves DOWNQUALITY for the current row.
			/// </summary>
			public double _DOWNQUALITY
			{
				get
				{
					return System.Convert.ToDouble(m_DR[10]);
				}
			}
			private static double [] a_UPPOSX = new double[ArraySize];
			/// <summary>
			/// Retrieves UPPOSX for the current row.
			/// </summary>
			public double _UPPOSX
			{
				get
				{
					return System.Convert.ToDouble(m_DR[11]);
				}
			}
			private static double [] a_UPPOSY = new double[ArraySize];
			/// <summary>
			/// Retrieves UPPOSY for the current row.
			/// </summary>
			public double _UPPOSY
			{
				get
				{
					return System.Convert.ToDouble(m_DR[12]);
				}
			}
			private static double [] a_UPPOSZ = new double[ArraySize];
			/// <summary>
			/// Retrieves UPPOSZ for the current row.
			/// </summary>
			public double _UPPOSZ
			{
				get
				{
					return System.Convert.ToDouble(m_DR[13]);
				}
			}
			private static double [] a_UPSLOPEX = new double[ArraySize];
			/// <summary>
			/// Retrieves UPSLOPEX for the current row.
			/// </summary>
			public double _UPSLOPEX
			{
				get
				{
					return System.Convert.ToDouble(m_DR[14]);
				}
			}
			private static double [] a_UPSLOPEY = new double[ArraySize];
			/// <summary>
			/// Retrieves UPSLOPEY for the current row.
			/// </summary>
			public double _UPSLOPEY
			{
				get
				{
					return System.Convert.ToDouble(m_DR[15]);
				}
			}
			private static double [] a_UPQUALITY = new double[ArraySize];
			/// <summary>
			/// Retrieves UPQUALITY for the current row.
			/// </summary>
			public double _UPQUALITY
			{
				get
				{
					return System.Convert.ToDouble(m_DR[16]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is actually inserted later, in a batch insertion command. The Flush method must be called to ensure all rows are actually written.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">the value to be inserted for ID_EVENTBRICK.</param>
			/// <param name="i_ID_RECONSTRUCTION">the value to be inserted for ID_RECONSTRUCTION.</param>
			/// <param name="i_ID_OPTION">the value to be inserted for ID_OPTION.</param>
			/// <param name="i_ID">the value to be inserted for ID.</param>
			/// <param name="i_TYPE">the value to be inserted for TYPE.</param>
			/// <param name="i_DOWNPOSX">the value to be inserted for DOWNPOSX.</param>
			/// <param name="i_DOWNPOSY">the value to be inserted for DOWNPOSY.</param>
			/// <param name="i_DOWNPOSZ">the value to be inserted for DOWNPOSZ.</param>
			/// <param name="i_DOWNSLOPEX">the value to be inserted for DOWNSLOPEX.</param>
			/// <param name="i_DOWNSLOPEY">the value to be inserted for DOWNSLOPEY.</param>
			/// <param name="i_DOWNQUALITY">the value to be inserted for DOWNQUALITY.</param>
			/// <param name="i_UPPOSX">the value to be inserted for UPPOSX.</param>
			/// <param name="i_UPPOSY">the value to be inserted for UPPOSY.</param>
			/// <param name="i_UPPOSZ">the value to be inserted for UPPOSZ.</param>
			/// <param name="i_UPSLOPEX">the value to be inserted for UPSLOPEX.</param>
			/// <param name="i_UPSLOPEY">the value to be inserted for UPSLOPEY.</param>
			/// <param name="i_UPQUALITY">the value to be inserted for UPQUALITY.</param>
			static public void Insert(long i_ID_EVENTBRICK,long i_ID_RECONSTRUCTION,long i_ID_OPTION,long i_ID,string i_TYPE,double i_DOWNPOSX,double i_DOWNPOSY,double i_DOWNPOSZ,double i_DOWNSLOPEX,double i_DOWNSLOPEY,double i_DOWNQUALITY,double i_UPPOSX,double i_UPPOSY,double i_UPPOSZ,double i_UPSLOPEX,double i_UPSLOPEY,double i_UPQUALITY)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				a_ID_EVENTBRICK[index] = i_ID_EVENTBRICK;
				a_ID_RECONSTRUCTION[index] = i_ID_RECONSTRUCTION;
				a_ID_OPTION[index] = i_ID_OPTION;
				a_ID[index] = i_ID;
				a_TYPE[index] = i_TYPE;
				a_DOWNPOSX[index] = i_DOWNPOSX;
				a_DOWNPOSY[index] = i_DOWNPOSY;
				a_DOWNPOSZ[index] = i_DOWNPOSZ;
				a_DOWNSLOPEX[index] = i_DOWNSLOPEX;
				a_DOWNSLOPEY[index] = i_DOWNSLOPEY;
				a_DOWNQUALITY[index] = i_DOWNQUALITY;
				a_UPPOSX[index] = i_UPPOSX;
				a_UPPOSY[index] = i_UPPOSY;
				a_UPPOSZ[index] = i_UPPOSZ;
				a_UPSLOPEX[index] = i_UPSLOPEX;
				a_UPSLOPEY[index] = i_UPSLOPEY;
				a_UPQUALITY[index] = i_UPQUALITY;
				if (++index >= ArraySize) { cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			}
			/// <summary>
			/// Reads a set of rows from TB_VOLUMETRACKS_FIT and retrieves them into a new TB_VOLUMETRACKS_FIT object.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_RECONSTRUCTION">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_OPTION">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_TYPE">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="order">the ordering scheme to be applied to returned rows. This applies to all columns in the primary key.</param>
			/// <returns>a new instance of the TB_VOLUMETRACKS_FIT class that can be used to read the retrieved data.</returns>
			static public TB_VOLUMETRACKS_FIT SelectPrimaryKey(object i_ID_EVENTBRICK,object i_ID_RECONSTRUCTION,object i_ID_OPTION,object i_ID,object i_TYPE, OrderBy order)
			{
				string wherestr = "";
				string wtempstr = "";
				if (i_ID_EVENTBRICK != null)
				{
					if (i_ID_EVENTBRICK == System.DBNull.Value) wtempstr = "ID_EVENTBRICK IS NULL";
					else wtempstr = "ID_EVENTBRICK = " + i_ID_EVENTBRICK.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_RECONSTRUCTION != null)
				{
					if (i_ID_RECONSTRUCTION == System.DBNull.Value) wtempstr = "ID_RECONSTRUCTION IS NULL";
					else wtempstr = "ID_RECONSTRUCTION = " + i_ID_RECONSTRUCTION.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_OPTION != null)
				{
					if (i_ID_OPTION == System.DBNull.Value) wtempstr = "ID_OPTION IS NULL";
					else wtempstr = "ID_OPTION = " + i_ID_OPTION.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID != null)
				{
					if (i_ID == System.DBNull.Value) wtempstr = "ID IS NULL";
					else wtempstr = "ID = " + i_ID.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_TYPE != null)
				{
					if (i_TYPE == System.DBNull.Value) wtempstr = "TYPE IS NULL";
					else wtempstr = "TYPE = " + i_TYPE.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (order == OrderBy.Ascending) return SelectWhere(wherestr, "ID_EVENTBRICK ASC,ID_RECONSTRUCTION ASC,ID_OPTION ASC,ID ASC,TYPE ASC");
				else if (order == OrderBy.Descending) return SelectWhere(wherestr, "ID_EVENTBRICK DESC,ID_RECONSTRUCTION DESC,ID_OPTION DESC,ID DESC,TYPE DESC");
				return SelectWhere(wherestr, null);
			}
			/// <summary>
			/// Reads a set of rows from TB_VOLUMETRACKS_FIT and retrieves them into a new TB_VOLUMETRACKS_FIT object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the TB_VOLUMETRACKS_FIT class that can be used to read the retrieved data.</returns>
			static public TB_VOLUMETRACKS_FIT SelectWhere(string wherestr, string orderstr)
			{
				TB_VOLUMETRACKS_FIT newobj = new TB_VOLUMETRACKS_FIT();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_EVENTBRICK,ID_RECONSTRUCTION,ID_OPTION,ID,TYPE,DOWNPOSX,DOWNPOSY,DOWNPOSZ,DOWNSLOPEX,DOWNSLOPEY,DOWNQUALITY,UPPOSX,UPPOSY,UPPOSZ,UPSLOPEX,UPSLOPEY,UPQUALITY FROM TB_VOLUMETRACKS_FIT" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO TB_VOLUMETRACKS_FIT (ID_EVENTBRICK,ID_RECONSTRUCTION,ID_OPTION,ID,TYPE,DOWNPOSX,DOWNPOSY,DOWNPOSZ,DOWNSLOPEX,DOWNSLOPEY,DOWNQUALITY,UPPOSX,UPPOSY,UPPOSZ,UPSLOPEX,UPSLOPEY,UPQUALITY) VALUES (:p_1,:p_2,:p_3,:p_4,:p_5,:p_6,:p_7,:p_8,:p_9,:p_10,:p_11,:p_12,:p_13,:p_14,:p_15,:p_16,:p_17)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_EVENTBRICK;
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_RECONSTRUCTION;
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_OPTION;
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID;
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input).Value = a_TYPE;
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_DOWNPOSX;
				newcmd.Parameters.Add("p_7", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_DOWNPOSY;
				newcmd.Parameters.Add("p_8", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_DOWNPOSZ;
				newcmd.Parameters.Add("p_9", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_DOWNSLOPEX;
				newcmd.Parameters.Add("p_10", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_DOWNSLOPEY;
				newcmd.Parameters.Add("p_11", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_DOWNQUALITY;
				newcmd.Parameters.Add("p_12", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_UPPOSX;
				newcmd.Parameters.Add("p_13", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_UPPOSY;
				newcmd.Parameters.Add("p_14", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_UPPOSZ;
				newcmd.Parameters.Add("p_15", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_UPSLOPEX;
				newcmd.Parameters.Add("p_16", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_UPSLOPEY;
				newcmd.Parameters.Add("p_17", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_UPQUALITY;
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the TB_VOLUME_SLICES table in the DB.
		/// For data insertion, the Insert and Flush static methods are used. This class inserts data in batches.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class TB_VOLUME_SLICES
		{
			internal TB_VOLUME_SLICES() {}
			System.Data.DataRowCollection m_DRC;
			const int ArraySize = 100;
			static int index = 0;
			/// <summary>
			/// Since this class uses batch insertion, this method must be called to ensure that all rows are actually inserted into TB_VOLUME_SLICES. Failure to do so will result in incomplete writes.
			/// </summary>
			static public void Flush() { if (cmd == null || index == 0) return; cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			private static long [] a_ID_EVENTBRICK = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_EVENTBRICK for the current row.
			/// </summary>
			public long _ID_EVENTBRICK
			{
				get
				{
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			private static long [] a_ID_VOLUME = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_VOLUME for the current row.
			/// </summary>
			public long _ID_VOLUME
			{
				get
				{
					return System.Convert.ToInt64(m_DR[1]);
				}
			}
			private static long [] a_ID_PLATE = new long[ArraySize];
			/// <summary>
			/// Retrieves ID_PLATE for the current row.
			/// </summary>
			public long _ID_PLATE
			{
				get
				{
					return System.Convert.ToInt64(m_DR[2]);
				}
			}
			private static double [] a_MINX = new double[ArraySize];
			/// <summary>
			/// Retrieves MINX for the current row.
			/// </summary>
			public double _MINX
			{
				get
				{
					return System.Convert.ToDouble(m_DR[3]);
				}
			}
			private static double [] a_MAXX = new double[ArraySize];
			/// <summary>
			/// Retrieves MAXX for the current row.
			/// </summary>
			public double _MAXX
			{
				get
				{
					return System.Convert.ToDouble(m_DR[4]);
				}
			}
			private static double [] a_MINY = new double[ArraySize];
			/// <summary>
			/// Retrieves MINY for the current row.
			/// </summary>
			public double _MINY
			{
				get
				{
					return System.Convert.ToDouble(m_DR[5]);
				}
			}
			private static double [] a_MAXY = new double[ArraySize];
			/// <summary>
			/// Retrieves MAXY for the current row.
			/// </summary>
			public double _MAXY
			{
				get
				{
					return System.Convert.ToDouble(m_DR[6]);
				}
			}
			private static object [] a_ID_ZONE = new object[ArraySize];
			/// <summary>
			/// Retrieves ID_ZONE for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _ID_ZONE
			{
				get
				{
					if (m_DR[7] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[7]);
				}
			}
			private static object [] a_DAMAGED = new object[ArraySize];
			/// <summary>
			/// Retrieves DAMAGED for the current row. The return value can be System.DBNull.Value or a value that can be cast to char.
			/// </summary>
			public object _DAMAGED
			{
				get
				{
					if (m_DR[8] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToChar(m_DR[8]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is actually inserted later, in a batch insertion command. The Flush method must be called to ensure all rows are actually written.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">the value to be inserted for ID_EVENTBRICK.</param>
			/// <param name="i_ID_VOLUME">the value to be inserted for ID_VOLUME.</param>
			/// <param name="i_ID_PLATE">the value to be inserted for ID_PLATE.</param>
			/// <param name="i_MINX">the value to be inserted for MINX.</param>
			/// <param name="i_MAXX">the value to be inserted for MAXX.</param>
			/// <param name="i_MINY">the value to be inserted for MINY.</param>
			/// <param name="i_MAXY">the value to be inserted for MAXY.</param>
			/// <param name="i_ID_ZONE">the value to be inserted for ID_ZONE. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_DAMAGED">the value to be inserted for DAMAGED. The value for this parameter can be char or System.DBNull.Value.</param>
			static public void Insert(long i_ID_EVENTBRICK,long i_ID_VOLUME,long i_ID_PLATE,double i_MINX,double i_MAXX,double i_MINY,double i_MAXY,object i_ID_ZONE,object i_DAMAGED)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				a_ID_EVENTBRICK[index] = i_ID_EVENTBRICK;
				a_ID_VOLUME[index] = i_ID_VOLUME;
				a_ID_PLATE[index] = i_ID_PLATE;
				a_MINX[index] = i_MINX;
				a_MAXX[index] = i_MAXX;
				a_MINY[index] = i_MINY;
				a_MAXY[index] = i_MAXY;
				a_ID_ZONE[index] = i_ID_ZONE;
				a_DAMAGED[index] = i_DAMAGED;
				if (++index >= ArraySize) { cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }
			}
			/// <summary>
			/// Reads a set of rows from TB_VOLUME_SLICES and retrieves them into a new TB_VOLUME_SLICES object.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_VOLUME">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID_PLATE">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="order">the ordering scheme to be applied to returned rows. This applies to all columns in the primary key.</param>
			/// <returns>a new instance of the TB_VOLUME_SLICES class that can be used to read the retrieved data.</returns>
			static public TB_VOLUME_SLICES SelectPrimaryKey(object i_ID_EVENTBRICK,object i_ID_VOLUME,object i_ID_PLATE, OrderBy order)
			{
				string wherestr = "";
				string wtempstr = "";
				if (i_ID_EVENTBRICK != null)
				{
					if (i_ID_EVENTBRICK == System.DBNull.Value) wtempstr = "ID_EVENTBRICK IS NULL";
					else wtempstr = "ID_EVENTBRICK = " + i_ID_EVENTBRICK.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_VOLUME != null)
				{
					if (i_ID_VOLUME == System.DBNull.Value) wtempstr = "ID_VOLUME IS NULL";
					else wtempstr = "ID_VOLUME = " + i_ID_VOLUME.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID_PLATE != null)
				{
					if (i_ID_PLATE == System.DBNull.Value) wtempstr = "ID_PLATE IS NULL";
					else wtempstr = "ID_PLATE = " + i_ID_PLATE.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (order == OrderBy.Ascending) return SelectWhere(wherestr, "ID_EVENTBRICK ASC,ID_VOLUME ASC,ID_PLATE ASC");
				else if (order == OrderBy.Descending) return SelectWhere(wherestr, "ID_EVENTBRICK DESC,ID_VOLUME DESC,ID_PLATE DESC");
				return SelectWhere(wherestr, null);
			}
			/// <summary>
			/// Reads a set of rows from TB_VOLUME_SLICES and retrieves them into a new TB_VOLUME_SLICES object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the TB_VOLUME_SLICES class that can be used to read the retrieved data.</returns>
			static public TB_VOLUME_SLICES SelectWhere(string wherestr, string orderstr)
			{
				TB_VOLUME_SLICES newobj = new TB_VOLUME_SLICES();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_EVENTBRICK,ID_VOLUME,ID_PLATE,MINX,MAXX,MINY,MAXY,ID_ZONE,DAMAGED FROM TB_VOLUME_SLICES" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO TB_VOLUME_SLICES (ID_EVENTBRICK,ID_VOLUME,ID_PLATE,MINX,MAXX,MINY,MAXY,ID_ZONE,DAMAGED) VALUES (:p_1,:p_2,:p_3,:p_4,:p_5,:p_6,:p_7,:p_8,:p_9)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_EVENTBRICK;
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_VOLUME;
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_PLATE;
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_MINX;
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_MAXX;
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_MINY;
				newcmd.Parameters.Add("p_7", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input).Value = a_MAXY;
				newcmd.Parameters.Add("p_8", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = a_ID_ZONE;
				newcmd.Parameters.Add("p_9", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input).Value = a_DAMAGED;
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the TB_ZONES table in the DB.
		/// For data insertion, the Insert method is used. Rows are inserted one by one.
		/// An instance of the class is produced for data retrieval.
		/// </summary>
		public class TB_ZONES
		{
			internal TB_ZONES() {}
			System.Data.DataRowCollection m_DRC;
			/// <summary>
			/// Retrieves ID_EVENTBRICK for the current row.
			/// </summary>
			public long _ID_EVENTBRICK
			{
				get
				{
					return System.Convert.ToInt64(m_DR[0]);
				}
			}
			/// <summary>
			/// Retrieves ID_PLATE for the current row.
			/// </summary>
			public long _ID_PLATE
			{
				get
				{
					return System.Convert.ToInt64(m_DR[1]);
				}
			}
			/// <summary>
			/// Retrieves ID_PROCESSOPERATION for the current row.
			/// </summary>
			public long _ID_PROCESSOPERATION
			{
				get
				{
					return System.Convert.ToInt64(m_DR[2]);
				}
			}
			/// <summary>
			/// Retrieves ID for the current row.
			/// </summary>
			public long _ID
			{
				get
				{
					return System.Convert.ToInt64(m_DR[3]);
				}
			}
			/// <summary>
			/// Retrieves MINX for the current row.
			/// </summary>
			public double _MINX
			{
				get
				{
					return System.Convert.ToDouble(m_DR[4]);
				}
			}
			/// <summary>
			/// Retrieves MAXX for the current row.
			/// </summary>
			public double _MAXX
			{
				get
				{
					return System.Convert.ToDouble(m_DR[5]);
				}
			}
			/// <summary>
			/// Retrieves MINY for the current row.
			/// </summary>
			public double _MINY
			{
				get
				{
					return System.Convert.ToDouble(m_DR[6]);
				}
			}
			/// <summary>
			/// Retrieves MAXY for the current row.
			/// </summary>
			public double _MAXY
			{
				get
				{
					return System.Convert.ToDouble(m_DR[7]);
				}
			}
			/// <summary>
			/// Retrieves RAWDATAPATH for the current row.
			/// </summary>
			public string _RAWDATAPATH
			{
				get
				{
					return System.Convert.ToString(m_DR[8]);
				}
			}
			/// <summary>
			/// Retrieves STARTTIME for the current row. The return value can be System.DBNull.Value or a value that can be cast to System.DateTime.
			/// </summary>
			public object _STARTTIME
			{
				get
				{
					if (m_DR[9] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDateTime(m_DR[9]);
				}
			}
			/// <summary>
			/// Retrieves ENDTIME for the current row. The return value can be System.DBNull.Value or a value that can be cast to System.DateTime.
			/// </summary>
			public object _ENDTIME
			{
				get
				{
					if (m_DR[10] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDateTime(m_DR[10]);
				}
			}
			/// <summary>
			/// Retrieves SERIES for the current row. The return value can be System.DBNull.Value or a value that can be cast to long.
			/// </summary>
			public object _SERIES
			{
				get
				{
					if (m_DR[11] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToInt64(m_DR[11]);
				}
			}
			/// <summary>
			/// Retrieves TXX for the current row. The return value can be System.DBNull.Value or a value that can be cast to double.
			/// </summary>
			public object _TXX
			{
				get
				{
					if (m_DR[12] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDouble(m_DR[12]);
				}
			}
			/// <summary>
			/// Retrieves TXY for the current row. The return value can be System.DBNull.Value or a value that can be cast to double.
			/// </summary>
			public object _TXY
			{
				get
				{
					if (m_DR[13] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDouble(m_DR[13]);
				}
			}
			/// <summary>
			/// Retrieves TYX for the current row. The return value can be System.DBNull.Value or a value that can be cast to double.
			/// </summary>
			public object _TYX
			{
				get
				{
					if (m_DR[14] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDouble(m_DR[14]);
				}
			}
			/// <summary>
			/// Retrieves TYY for the current row. The return value can be System.DBNull.Value or a value that can be cast to double.
			/// </summary>
			public object _TYY
			{
				get
				{
					if (m_DR[15] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDouble(m_DR[15]);
				}
			}
			/// <summary>
			/// Retrieves TDX for the current row. The return value can be System.DBNull.Value or a value that can be cast to double.
			/// </summary>
			public object _TDX
			{
				get
				{
					if (m_DR[16] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDouble(m_DR[16]);
				}
			}
			/// <summary>
			/// Retrieves TDY for the current row. The return value can be System.DBNull.Value or a value that can be cast to double.
			/// </summary>
			public object _TDY
			{
				get
				{
					if (m_DR[17] == System.DBNull.Value) return System.DBNull.Value;
					return System.Convert.ToDouble(m_DR[17]);
				}
			}
			/// <summary>
			/// Inserts one row into the DB. The row is inserted immediately.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">the value to be inserted for ID_EVENTBRICK.</param>
			/// <param name="i_ID_PLATE">the value to be inserted for ID_PLATE.</param>
			/// <param name="i_ID_PROCESSOPERATION">the value to be inserted for ID_PROCESSOPERATION.</param>
			/// <param name="i_ID">the value to be inserted for ID. This value is actually used only if this method call is involved in data publication/replication, otherwise the actual value is generated by the OPERA DB and the supplied value is ignored.</param>
			/// <param name="i_MINX">the value to be inserted for MINX.</param>
			/// <param name="i_MAXX">the value to be inserted for MAXX.</param>
			/// <param name="i_MINY">the value to be inserted for MINY.</param>
			/// <param name="i_MAXY">the value to be inserted for MAXY.</param>
			/// <param name="i_RAWDATAPATH">the value to be inserted for RAWDATAPATH.</param>
			/// <param name="i_STARTTIME">the value to be inserted for STARTTIME. The value for this parameter can be System.DateTime or System.DBNull.Value.</param>
			/// <param name="i_ENDTIME">the value to be inserted for ENDTIME. The value for this parameter can be System.DateTime or System.DBNull.Value.</param>
			/// <param name="i_SERIES">the value to be inserted for SERIES. The value for this parameter can be long or System.DBNull.Value.</param>
			/// <param name="i_TXX">the value to be inserted for TXX. The value for this parameter can be double or System.DBNull.Value.</param>
			/// <param name="i_TXY">the value to be inserted for TXY. The value for this parameter can be double or System.DBNull.Value.</param>
			/// <param name="i_TYX">the value to be inserted for TYX. The value for this parameter can be double or System.DBNull.Value.</param>
			/// <param name="i_TYY">the value to be inserted for TYY. The value for this parameter can be double or System.DBNull.Value.</param>
			/// <param name="i_TDX">the value to be inserted for TDX. The value for this parameter can be double or System.DBNull.Value.</param>
			/// <param name="i_TDY">the value to be inserted for TDY. The value for this parameter can be double or System.DBNull.Value.</param>
			/// <returns>the value of ID for the new row.</returns>
			static public long Insert(long i_ID_EVENTBRICK,long i_ID_PLATE,long i_ID_PROCESSOPERATION,long i_ID,double i_MINX,double i_MAXX,double i_MINY,double i_MAXY,string i_RAWDATAPATH,object i_STARTTIME,object i_ENDTIME,object i_SERIES,object i_TXX,object i_TXY,object i_TYX,object i_TYY,object i_TDX,object i_TDY)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = i_ID_EVENTBRICK;
				cmd.Parameters[1].Value = i_ID_PLATE;
				cmd.Parameters[2].Value = i_ID_PROCESSOPERATION;
				cmd.Parameters[3].Value = i_ID;
				cmd.Parameters[4].Value = i_MINX;
				cmd.Parameters[5].Value = i_MAXX;
				cmd.Parameters[6].Value = i_MINY;
				cmd.Parameters[7].Value = i_MAXY;
				cmd.Parameters[8].Value = i_RAWDATAPATH;
				cmd.Parameters[9].Value = i_STARTTIME;
				cmd.Parameters[10].Value = i_ENDTIME;
				cmd.Parameters[11].Value = i_SERIES;
				cmd.Parameters[12].Value = i_TXX;
				cmd.Parameters[13].Value = i_TXY;
				cmd.Parameters[14].Value = i_TYX;
				cmd.Parameters[15].Value = i_TYY;
				cmd.Parameters[16].Value = i_TDX;
				cmd.Parameters[17].Value = i_TDY;
				cmd.ExecuteNonQuery();
				return SySal.OperaDb.Convert.ToInt64(cmd.Parameters[18].Value);
			}
			/// <summary>
			/// Reads a set of rows from TB_ZONES and retrieves them into a new TB_ZONES object.
			/// </summary>
			/// <param name="i_ID_EVENTBRICK">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="i_ID">if non-null, only rows that have this field equal to the specified value are returned.</param>
			/// <param name="order">the ordering scheme to be applied to returned rows. This applies to all columns in the primary key.</param>
			/// <returns>a new instance of the TB_ZONES class that can be used to read the retrieved data.</returns>
			static public TB_ZONES SelectPrimaryKey(object i_ID_EVENTBRICK,object i_ID, OrderBy order)
			{
				string wherestr = "";
				string wtempstr = "";
				if (i_ID_EVENTBRICK != null)
				{
					if (i_ID_EVENTBRICK == System.DBNull.Value) wtempstr = "ID_EVENTBRICK IS NULL";
					else wtempstr = "ID_EVENTBRICK = " + i_ID_EVENTBRICK.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (i_ID != null)
				{
					if (i_ID == System.DBNull.Value) wtempstr = "ID IS NULL";
					else wtempstr = "ID = " + i_ID.ToString();
					if (wherestr.Length == 0) wherestr = wtempstr; else wherestr += " AND " + wtempstr;
				}
				if (order == OrderBy.Ascending) return SelectWhere(wherestr, "ID_EVENTBRICK ASC,ID ASC");
				else if (order == OrderBy.Descending) return SelectWhere(wherestr, "ID_EVENTBRICK DESC,ID DESC");
				return SelectWhere(wherestr, null);
			}
			/// <summary>
			/// Reads a set of rows from TB_ZONES and retrieves them into a new TB_ZONES object.
			/// </summary>
			/// <param name="wherestr">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>
			/// <param name="orderstr">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>
			/// <returns>a new instance of the TB_ZONES class that can be used to read the retrieved data.</returns>
			static public TB_ZONES SelectWhere(string wherestr, string orderstr)
			{
				TB_ZONES newobj = new TB_ZONES();
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_EVENTBRICK,ID_PLATE,ID_PROCESSOPERATION,ID,MINX,MAXX,MINY,MAXY,RAWDATAPATH,STARTTIME,ENDTIME,SERIES,TXX,TXY,TYX,TYY,TDX,TDY FROM TB_ZONES" + ((wherestr == null || wherestr.Trim().Length == 0) ? "" : (" WHERE(" + wherestr + ")" + ((orderstr != null && orderstr.Trim() != "") ? (" ORDER BY " + orderstr): "") )), Schema.m_DB).Fill(ds);
				newobj.m_DRC = ds.Tables[0].Rows;
				newobj.m_Row = -1;
				return newobj;
			}
			/// <summary>
			/// the number of rows retrieved.
			/// </summary>
			public int Count { get { return m_DRC.Count; } }
			internal int m_Row;
			internal System.Data.DataRow m_DR;
			/// <summary>
			/// the current row for which field values are exposed.
			/// </summary>
			public int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO TB_ZONES (ID_EVENTBRICK,ID_PLATE,ID_PROCESSOPERATION,ID,MINX,MAXX,MINY,MAXY,RAWDATAPATH,STARTTIME,ENDTIME,SERIES,TXX,TXY,TYX,TYY,TDX,TDY) VALUES (:p_1,:p_2,:p_3,:p_4,:p_5,:p_6,:p_7,:p_8,:p_9,:p_10,:p_11,:p_12,:p_13,:p_14,:p_15,:p_16,:p_17,:p_18) RETURNING ID INTO :o_18");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_7", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_8", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_9", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_10", SySal.OperaDb.OperaDbType.DateTime, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_11", SySal.OperaDb.OperaDbType.DateTime, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_12", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_13", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_14", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_15", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_16", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_17", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_18", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("o_18", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Output);
				return newcmd;
			}
		}
		/// <summary>
		/// Accesses the LP_ADD_PROC_OPERATION procedure in the DB.
		/// </summary>
		public class LP_ADD_PROC_OPERATION
		{
			private LP_ADD_PROC_OPERATION() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL LP_ADD_PROC_OPERATION(:p_1,:p_2,:p_3,:p_4,:p_5,:p_6,:p_7,:p_8,:p_9,:p_10)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Output);
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Output);
				newcmd.Parameters.Add("p_7", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_8", SySal.OperaDb.OperaDbType.DateTime, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_9", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_10", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Output);
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			/// <param name = "O_MACHINEID">the value of O_MACHINEID to be used for the procedure call.</param>
			/// <param name = "O_PROGSETID">the value of O_PROGSETID to be used for the procedure call.</param>
			/// <param name = "O_USR">the value of O_USR to be used for the procedure call.</param>
			/// <param name = "O_PWD">the value of O_PWD to be used for the procedure call.</param>
			/// <param name = "O_TOKEN">the value of O_TOKEN obtained from the procedure call as an output. If null, the output is ignored.</param>
			/// <param name = "O_REQID">the value of O_REQID obtained from the procedure call as an output. If null, the output is ignored.</param>
			/// <param name = "O_PARENTOPID">the value of O_PARENTOPID to be used for the procedure call.</param>
			/// <param name = "O_STARTTIME">the value of O_STARTTIME to be used for the procedure call.</param>
			/// <param name = "O_NOTES">the value of O_NOTES to be used for the procedure call.</param>
			/// <param name = "NEWID">the value of NEWID obtained from the procedure call as an output. If null, the output is ignored.</param>
			public static void Call(object O_MACHINEID, object O_PROGSETID, object O_USR, object O_PWD, ref object O_TOKEN, ref object O_REQID, object O_PARENTOPID, object O_STARTTIME, object O_NOTES, ref object NEWID)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = O_MACHINEID;
				cmd.Parameters[1].Value = O_PROGSETID;
				cmd.Parameters[2].Value = O_USR;
				cmd.Parameters[3].Value = O_PWD;
				cmd.Parameters[6].Value = O_PARENTOPID;
				cmd.Parameters[7].Value = O_STARTTIME;
				cmd.Parameters[8].Value = O_NOTES;
				cmd.ExecuteNonQuery();
				if (O_TOKEN != null) O_TOKEN = cmd.Parameters[4].Value;
				if (O_REQID != null) O_REQID = cmd.Parameters[5].Value;
				if (NEWID != null) NEWID = cmd.Parameters[9].Value;
			}
		}
		/// <summary>
		/// Accesses the LP_ADD_PROC_OPERATION_BRICK procedure in the DB.
		/// </summary>
		public class LP_ADD_PROC_OPERATION_BRICK
		{
			private LP_ADD_PROC_OPERATION_BRICK() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL LP_ADD_PROC_OPERATION_BRICK(:p_1,:p_2,:p_3,:p_4,:p_5,:p_6,:p_7,:p_8,:p_9,:p_10,:p_11)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Output);
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Output);
				newcmd.Parameters.Add("p_7", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_8", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_9", SySal.OperaDb.OperaDbType.DateTime, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_10", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_11", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Output);
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			/// <param name = "O_MACHINEID">the value of O_MACHINEID to be used for the procedure call.</param>
			/// <param name = "O_PROGSETID">the value of O_PROGSETID to be used for the procedure call.</param>
			/// <param name = "O_USR">the value of O_USR to be used for the procedure call.</param>
			/// <param name = "O_PWD">the value of O_PWD to be used for the procedure call.</param>
			/// <param name = "O_TOKEN">the value of O_TOKEN obtained from the procedure call as an output. If null, the output is ignored.</param>
			/// <param name = "O_REQID">the value of O_REQID obtained from the procedure call as an output. If null, the output is ignored.</param>
			/// <param name = "O_BRICKID">the value of O_BRICKID to be used for the procedure call.</param>
			/// <param name = "O_PARENTOPID">the value of O_PARENTOPID to be used for the procedure call.</param>
			/// <param name = "O_STARTTIME">the value of O_STARTTIME to be used for the procedure call.</param>
			/// <param name = "O_NOTES">the value of O_NOTES to be used for the procedure call.</param>
			/// <param name = "NEWID">the value of NEWID obtained from the procedure call as an output. If null, the output is ignored.</param>
			public static void Call(object O_MACHINEID, object O_PROGSETID, object O_USR, object O_PWD, ref object O_TOKEN, ref object O_REQID, object O_BRICKID, object O_PARENTOPID, object O_STARTTIME, object O_NOTES, ref object NEWID)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = O_MACHINEID;
				cmd.Parameters[1].Value = O_PROGSETID;
				cmd.Parameters[2].Value = O_USR;
				cmd.Parameters[3].Value = O_PWD;
				cmd.Parameters[6].Value = O_BRICKID;
				cmd.Parameters[7].Value = O_PARENTOPID;
				cmd.Parameters[8].Value = O_STARTTIME;
				cmd.Parameters[9].Value = O_NOTES;
				cmd.ExecuteNonQuery();
				if (O_TOKEN != null) O_TOKEN = cmd.Parameters[4].Value;
				if (O_REQID != null) O_REQID = cmd.Parameters[5].Value;
				if (NEWID != null) NEWID = cmd.Parameters[10].Value;
			}
		}
		/// <summary>
		/// Accesses the LP_ADD_PROC_OPERATION_PLATE procedure in the DB.
		/// </summary>
		public class LP_ADD_PROC_OPERATION_PLATE
		{
			private LP_ADD_PROC_OPERATION_PLATE() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL LP_ADD_PROC_OPERATION_PLATE(:p_1,:p_2,:p_3,:p_4,:p_5,:p_6,:p_7,:p_8,:p_9,:p_10,:p_11,:p_12,:p_13)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Output);
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Output);
				newcmd.Parameters.Add("p_7", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_8", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_9", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_10", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_11", SySal.OperaDb.OperaDbType.DateTime, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_12", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_13", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Output);
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			/// <param name = "O_MACHINEID">the value of O_MACHINEID to be used for the procedure call.</param>
			/// <param name = "O_PROGSETID">the value of O_PROGSETID to be used for the procedure call.</param>
			/// <param name = "O_USR">the value of O_USR to be used for the procedure call.</param>
			/// <param name = "O_PWD">the value of O_PWD to be used for the procedure call.</param>
			/// <param name = "O_TOKEN">the value of O_TOKEN obtained from the procedure call as an output. If null, the output is ignored.</param>
			/// <param name = "O_REQID">the value of O_REQID obtained from the procedure call as an output. If null, the output is ignored.</param>
			/// <param name = "O_BRICKID">the value of O_BRICKID to be used for the procedure call.</param>
			/// <param name = "O_PLATEID">the value of O_PLATEID to be used for the procedure call.</param>
			/// <param name = "O_PARENTOPID">the value of O_PARENTOPID to be used for the procedure call.</param>
			/// <param name = "O_CALIBRATION">the value of O_CALIBRATION to be used for the procedure call.</param>
			/// <param name = "O_STARTTIME">the value of O_STARTTIME to be used for the procedure call.</param>
			/// <param name = "O_NOTES">the value of O_NOTES to be used for the procedure call.</param>
			/// <param name = "NEWID">the value of NEWID obtained from the procedure call as an output. If null, the output is ignored.</param>
			public static void Call(object O_MACHINEID, object O_PROGSETID, object O_USR, object O_PWD, ref object O_TOKEN, ref object O_REQID, object O_BRICKID, object O_PLATEID, object O_PARENTOPID, object O_CALIBRATION, object O_STARTTIME, object O_NOTES, ref object NEWID)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = O_MACHINEID;
				cmd.Parameters[1].Value = O_PROGSETID;
				cmd.Parameters[2].Value = O_USR;
				cmd.Parameters[3].Value = O_PWD;
				cmd.Parameters[6].Value = O_BRICKID;
				cmd.Parameters[7].Value = O_PLATEID;
				cmd.Parameters[8].Value = O_PARENTOPID;
				cmd.Parameters[9].Value = O_CALIBRATION;
				cmd.Parameters[10].Value = O_STARTTIME;
				cmd.Parameters[11].Value = O_NOTES;
				cmd.ExecuteNonQuery();
				if (O_TOKEN != null) O_TOKEN = cmd.Parameters[4].Value;
				if (O_REQID != null) O_REQID = cmd.Parameters[5].Value;
				if (NEWID != null) NEWID = cmd.Parameters[12].Value;
			}
		}
		/// <summary>
		/// Accesses the LP_CHECK_ACCESS procedure in the DB.
		/// </summary>
		public class LP_CHECK_ACCESS
		{
			private LP_CHECK_ACCESS() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL LP_CHECK_ACCESS(:p_1,:p_2,:p_3,:p_4,:p_5,:p_6,:p_7)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_7", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			/// <param name = "O_TOKEN">the value of O_TOKEN to be used for the procedure call.</param>
			/// <param name = "O_REQUESTSCAN">the value of O_REQUESTSCAN to be used for the procedure call.</param>
			/// <param name = "O_REQUESTWEBANALYSIS">the value of O_REQUESTWEBANALYSIS to be used for the procedure call.</param>
			/// <param name = "O_REQUESTDATADOWNLOAD">the value of O_REQUESTDATADOWNLOAD to be used for the procedure call.</param>
			/// <param name = "O_REQUESTDATAPROCESSING">the value of O_REQUESTDATAPROCESSING to be used for the procedure call.</param>
			/// <param name = "O_REQUESTPROCESSSTARTUP">the value of O_REQUESTPROCESSSTARTUP to be used for the procedure call.</param>
			/// <param name = "O_ADMINISTER">the value of O_ADMINISTER to be used for the procedure call.</param>
			public static void Call(object O_TOKEN, object O_REQUESTSCAN, object O_REQUESTWEBANALYSIS, object O_REQUESTDATADOWNLOAD, object O_REQUESTDATAPROCESSING, object O_REQUESTPROCESSSTARTUP, object O_ADMINISTER)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = O_TOKEN;
				cmd.Parameters[1].Value = O_REQUESTSCAN;
				cmd.Parameters[2].Value = O_REQUESTWEBANALYSIS;
				cmd.Parameters[3].Value = O_REQUESTDATADOWNLOAD;
				cmd.Parameters[4].Value = O_REQUESTDATAPROCESSING;
				cmd.Parameters[5].Value = O_REQUESTPROCESSSTARTUP;
				cmd.Parameters[6].Value = O_ADMINISTER;
				cmd.ExecuteNonQuery();
			}
		}
		/// <summary>
		/// Accesses the LP_CHECK_TOKEN_OWNERSHIP procedure in the DB.
		/// </summary>
		public class LP_CHECK_TOKEN_OWNERSHIP
		{
			private LP_CHECK_TOKEN_OWNERSHIP() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL LP_CHECK_TOKEN_OWNERSHIP(:p_1,:p_2,:p_3,:p_4)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.InputOutput);
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			/// <param name = "O_TOKEN">the value of O_TOKEN to be used for the procedure call.</param>
			/// <param name = "O_UID">the value of O_UID to be used for the procedure call, which can be modified on procedure completion as an output. If null, the input is replaced with a <c>System.DBNull.Value</c> and the output is ignored.</param>
			/// <param name = "O_USR">the value of O_USR to be used for the procedure call.</param>
			/// <param name = "O_PWD">the value of O_PWD to be used for the procedure call.</param>
			public static void Call(object O_TOKEN, object O_UID, object O_USR, object O_PWD)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = O_TOKEN;
				if (O_UID == null) cmd.Parameters[1].Value = System.DBNull.Value; else cmd.Parameters[1].Value = O_UID;
				cmd.Parameters[2].Value = O_USR;
				cmd.Parameters[3].Value = O_PWD;
				cmd.ExecuteNonQuery();
				if (O_UID != null) O_UID = cmd.Parameters[1].Value;
			}
		}
		/// <summary>
		/// Accesses the LP_CLEAN_ORPHAN_TOKENS procedure in the DB.
		/// </summary>
		public class LP_CLEAN_ORPHAN_TOKENS
		{
			private LP_CLEAN_ORPHAN_TOKENS() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL LP_CLEAN_ORPHAN_TOKENS()");
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			public static void Call()
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.ExecuteNonQuery();
			}
		}
		/// <summary>
		/// Accesses the LP_DATABUFFER_FLUSH procedure in the DB.
		/// </summary>
		public class LP_DATABUFFER_FLUSH
		{
			private LP_DATABUFFER_FLUSH() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL LP_DATABUFFER_FLUSH()");
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			public static void Call()
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.ExecuteNonQuery();
			}
		}
		/// <summary>
		/// Accesses the LP_FAIL_OPERATION procedure in the DB.
		/// </summary>
		public class LP_FAIL_OPERATION
		{
			private LP_FAIL_OPERATION() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL LP_FAIL_OPERATION(:p_1,:p_2)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.DateTime, System.Data.ParameterDirection.Input);
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			/// <param name = "O_ID">the value of O_ID to be used for the procedure call.</param>
			/// <param name = "O_FINISHTIME">the value of O_FINISHTIME to be used for the procedure call.</param>
			public static void Call(object O_ID, object O_FINISHTIME)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = O_ID;
				cmd.Parameters[1].Value = O_FINISHTIME;
				cmd.ExecuteNonQuery();
			}
		}
		/// <summary>
		/// Accesses the LP_SUCCESS_OPERATION procedure in the DB.
		/// </summary>
		public class LP_SUCCESS_OPERATION
		{
			private LP_SUCCESS_OPERATION() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL LP_SUCCESS_OPERATION(:p_1,:p_2)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.DateTime, System.Data.ParameterDirection.Input);
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			/// <param name = "O_ID">the value of O_ID to be used for the procedure call.</param>
			/// <param name = "O_FINISHTIME">the value of O_FINISHTIME to be used for the procedure call.</param>
			public static void Call(object O_ID, object O_FINISHTIME)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = O_ID;
				cmd.Parameters[1].Value = O_FINISHTIME;
				cmd.ExecuteNonQuery();
			}
		}
		/// <summary>
		/// Accesses the PC_ADD_BRICK procedure in the DB.
		/// </summary>
		public class PC_ADD_BRICK
		{
			private PC_ADD_BRICK() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL PC_ADD_BRICK(:p_1,:p_2,:p_3,:p_4,:p_5,:p_6,:p_7,:p_8,:p_9)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_7", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_8", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_9", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Output);
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			/// <param name = "B_IDBRICK">the value of B_IDBRICK to be used for the procedure call.</param>
			/// <param name = "B_MINX">the value of B_MINX to be used for the procedure call.</param>
			/// <param name = "B_MAXX">the value of B_MAXX to be used for the procedure call.</param>
			/// <param name = "B_MINY">the value of B_MINY to be used for the procedure call.</param>
			/// <param name = "B_MAXY">the value of B_MAXY to be used for the procedure call.</param>
			/// <param name = "B_MINZ">the value of B_MINZ to be used for the procedure call.</param>
			/// <param name = "B_MAXZ">the value of B_MAXZ to be used for the procedure call.</param>
			/// <param name = "B_SET">the value of B_SET to be used for the procedure call.</param>
			/// <param name = "B_ID">the value of B_ID obtained from the procedure call as an output. If null, the output is ignored.</param>
			public static void Call(object B_IDBRICK, object B_MINX, object B_MAXX, object B_MINY, object B_MAXY, object B_MINZ, object B_MAXZ, object B_SET, ref object B_ID)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = B_IDBRICK;
				cmd.Parameters[1].Value = B_MINX;
				cmd.Parameters[2].Value = B_MAXX;
				cmd.Parameters[3].Value = B_MINY;
				cmd.Parameters[4].Value = B_MAXY;
				cmd.Parameters[5].Value = B_MINZ;
				cmd.Parameters[6].Value = B_MAXZ;
				cmd.Parameters[7].Value = B_SET;
				cmd.ExecuteNonQuery();
				if (B_ID != null) B_ID = cmd.Parameters[8].Value;
			}
		}
		/// <summary>
		/// Accesses the PC_ADD_BRICK_EASY procedure in the DB.
		/// </summary>
		public class PC_ADD_BRICK_EASY
		{
			private PC_ADD_BRICK_EASY() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL PC_ADD_BRICK_EASY(:p_1,:p_2,:p_3,:p_4,:p_5,:p_6,:p_7,:p_8,:p_9,:p_10)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_7", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_8", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_9", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_10", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Output);
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			/// <param name = "B_IDBRICK">the value of B_IDBRICK to be used for the procedure call.</param>
			/// <param name = "B_MINX">the value of B_MINX to be used for the procedure call.</param>
			/// <param name = "B_MAXX">the value of B_MAXX to be used for the procedure call.</param>
			/// <param name = "B_MINY">the value of B_MINY to be used for the procedure call.</param>
			/// <param name = "B_MAXY">the value of B_MAXY to be used for the procedure call.</param>
			/// <param name = "B_NPL">the value of B_NPL to be used for the procedure call.</param>
			/// <param name = "B_MAXZ">the value of B_MAXZ to be used for the procedure call.</param>
			/// <param name = "B_DOWN">the value of B_DOWN to be used for the procedure call.</param>
			/// <param name = "B_SET">the value of B_SET to be used for the procedure call.</param>
			/// <param name = "B_ID">the value of B_ID obtained from the procedure call as an output. If null, the output is ignored.</param>
			public static void Call(object B_IDBRICK, object B_MINX, object B_MAXX, object B_MINY, object B_MAXY, object B_NPL, object B_MAXZ, object B_DOWN, object B_SET, ref object B_ID)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = B_IDBRICK;
				cmd.Parameters[1].Value = B_MINX;
				cmd.Parameters[2].Value = B_MAXX;
				cmd.Parameters[3].Value = B_MINY;
				cmd.Parameters[4].Value = B_MAXY;
				cmd.Parameters[5].Value = B_NPL;
				cmd.Parameters[6].Value = B_MAXZ;
				cmd.Parameters[7].Value = B_DOWN;
				cmd.Parameters[8].Value = B_SET;
				cmd.ExecuteNonQuery();
				if (B_ID != null) B_ID = cmd.Parameters[9].Value;
			}
		}
		/// <summary>
		/// Accesses the PC_ADD_BRICK_EASY_Z procedure in the DB.
		/// </summary>
		public class PC_ADD_BRICK_EASY_Z
		{
			private PC_ADD_BRICK_EASY_Z() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL PC_ADD_BRICK_EASY_Z(:p_1,:p_2,:p_3,:p_4,:p_5,:p_6,:p_7,:p_8,:p_9,:p_10,:p_11,:p_12)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_7", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_8", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_9", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_10", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Output);
				newcmd.Parameters.Add("p_11", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_12", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			/// <param name = "B_IDBRICK">the value of B_IDBRICK to be used for the procedure call.</param>
			/// <param name = "B_MINX">the value of B_MINX to be used for the procedure call.</param>
			/// <param name = "B_MAXX">the value of B_MAXX to be used for the procedure call.</param>
			/// <param name = "B_MINY">the value of B_MINY to be used for the procedure call.</param>
			/// <param name = "B_MAXY">the value of B_MAXY to be used for the procedure call.</param>
			/// <param name = "B_NPL">the value of B_NPL to be used for the procedure call.</param>
			/// <param name = "B_MAXZ">the value of B_MAXZ to be used for the procedure call.</param>
			/// <param name = "B_DOWN">the value of B_DOWN to be used for the procedure call.</param>
			/// <param name = "B_SET">the value of B_SET to be used for the procedure call.</param>
			/// <param name = "B_ID">the value of B_ID obtained from the procedure call as an output. If null, the output is ignored.</param>
			/// <param name = "B_ZEROX">the value of B_ZEROX to be used for the procedure call.</param>
			/// <param name = "B_ZEROY">the value of B_ZEROY to be used for the procedure call.</param>
			public static void Call(object B_IDBRICK, object B_MINX, object B_MAXX, object B_MINY, object B_MAXY, object B_NPL, object B_MAXZ, object B_DOWN, object B_SET, ref object B_ID, object B_ZEROX, object B_ZEROY)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = B_IDBRICK;
				cmd.Parameters[1].Value = B_MINX;
				cmd.Parameters[2].Value = B_MAXX;
				cmd.Parameters[3].Value = B_MINY;
				cmd.Parameters[4].Value = B_MAXY;
				cmd.Parameters[5].Value = B_NPL;
				cmd.Parameters[6].Value = B_MAXZ;
				cmd.Parameters[7].Value = B_DOWN;
				cmd.Parameters[8].Value = B_SET;
				cmd.Parameters[10].Value = B_ZEROX;
				cmd.Parameters[11].Value = B_ZEROY;
				cmd.ExecuteNonQuery();
				if (B_ID != null) B_ID = cmd.Parameters[9].Value;
			}
		}
		/// <summary>
		/// Accesses the PC_ADD_BRICK_SET procedure in the DB.
		/// </summary>
		public class PC_ADD_BRICK_SET
		{
			private PC_ADD_BRICK_SET() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL PC_ADD_BRICK_SET(:p_1,:p_2,:p_3,:p_4,:p_5)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			/// <param name = "NEWID">the value of NEWID to be used for the procedure call.</param>
			/// <param name = "RNGMIN">the value of RNGMIN to be used for the procedure call.</param>
			/// <param name = "RNGMAX">the value of RNGMAX to be used for the procedure call.</param>
			/// <param name = "TBLSPC_EXT">the value of TBLSPC_EXT to be used for the procedure call.</param>
			/// <param name = "DEFID">the value of DEFID to be used for the procedure call.</param>
			public static void Call(object NEWID, object RNGMIN, object RNGMAX, object TBLSPC_EXT, object DEFID)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = NEWID;
				cmd.Parameters[1].Value = RNGMIN;
				cmd.Parameters[2].Value = RNGMAX;
				cmd.Parameters[3].Value = TBLSPC_EXT;
				cmd.Parameters[4].Value = DEFID;
				cmd.ExecuteNonQuery();
			}
		}
		/// <summary>
		/// Accesses the PC_ADD_BRICK_SPACE procedure in the DB.
		/// </summary>
		public class PC_ADD_BRICK_SPACE
		{
			private PC_ADD_BRICK_SPACE() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL PC_ADD_BRICK_SPACE(:p_1,:p_2)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			/// <param name = "NEWID">the value of NEWID to be used for the procedure call.</param>
			/// <param name = "IDSET">the value of IDSET to be used for the procedure call.</param>
			public static void Call(object NEWID, object IDSET)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = NEWID;
				cmd.Parameters[1].Value = IDSET;
				cmd.ExecuteNonQuery();
			}
		}
		/// <summary>
		/// Accesses the PC_ADD_CS_DOUBLET procedure in the DB.
		/// </summary>
		public class PC_ADD_CS_DOUBLET
		{
			private PC_ADD_CS_DOUBLET() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL PC_ADD_CS_DOUBLET(:p_1,:p_2,:p_3,:p_4,:p_5,:p_6,:p_7,:p_8,:p_9)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_7", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_8", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_9", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Output);
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			/// <param name = "B_IDBRICK">the value of B_IDBRICK to be used for the procedure call.</param>
			/// <param name = "B_MINX">the value of B_MINX to be used for the procedure call.</param>
			/// <param name = "B_MAXX">the value of B_MAXX to be used for the procedure call.</param>
			/// <param name = "B_MINY">the value of B_MINY to be used for the procedure call.</param>
			/// <param name = "B_MAXY">the value of B_MAXY to be used for the procedure call.</param>
			/// <param name = "B_MINZ">the value of B_MINZ to be used for the procedure call.</param>
			/// <param name = "B_MAXZ">the value of B_MAXZ to be used for the procedure call.</param>
			/// <param name = "B_SET">the value of B_SET to be used for the procedure call.</param>
			/// <param name = "B_ID">the value of B_ID obtained from the procedure call as an output. If null, the output is ignored.</param>
			public static void Call(object B_IDBRICK, object B_MINX, object B_MAXX, object B_MINY, object B_MAXY, object B_MINZ, object B_MAXZ, object B_SET, ref object B_ID)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = B_IDBRICK;
				cmd.Parameters[1].Value = B_MINX;
				cmd.Parameters[2].Value = B_MAXX;
				cmd.Parameters[3].Value = B_MINY;
				cmd.Parameters[4].Value = B_MAXY;
				cmd.Parameters[5].Value = B_MINZ;
				cmd.Parameters[6].Value = B_MAXZ;
				cmd.Parameters[7].Value = B_SET;
				cmd.ExecuteNonQuery();
				if (B_ID != null) B_ID = cmd.Parameters[8].Value;
			}
		}
		/// <summary>
		/// Accesses the PC_ADD_MACHINE procedure in the DB.
		/// </summary>
		public class PC_ADD_MACHINE
		{
			private PC_ADD_MACHINE() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL PC_ADD_MACHINE(:p_1,:p_2,:p_3,:p_4,:p_5,:p_6,:p_7,:p_8,:p_9)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_7", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_8", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_9", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Output);
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			/// <param name = "S_ID">the value of S_ID to be used for the procedure call.</param>
			/// <param name = "M_NAME">the value of M_NAME to be used for the procedure call.</param>
			/// <param name = "M_ADDRESS">the value of M_ADDRESS to be used for the procedure call.</param>
			/// <param name = "M_SCANSRV">the value of M_SCANSRV to be used for the procedure call.</param>
			/// <param name = "M_BATCHSRV">the value of M_BATCHSRV to be used for the procedure call.</param>
			/// <param name = "M_DATAPROCSRV">the value of M_DATAPROCSRV to be used for the procedure call.</param>
			/// <param name = "M_WEBSRV">the value of M_WEBSRV to be used for the procedure call.</param>
			/// <param name = "M_DBSRV">the value of M_DBSRV to be used for the procedure call.</param>
			/// <param name = "NEWID">the value of NEWID obtained from the procedure call as an output. If null, the output is ignored.</param>
			public static void Call(object S_ID, object M_NAME, object M_ADDRESS, object M_SCANSRV, object M_BATCHSRV, object M_DATAPROCSRV, object M_WEBSRV, object M_DBSRV, ref object NEWID)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = S_ID;
				cmd.Parameters[1].Value = M_NAME;
				cmd.Parameters[2].Value = M_ADDRESS;
				cmd.Parameters[3].Value = M_SCANSRV;
				cmd.Parameters[4].Value = M_BATCHSRV;
				cmd.Parameters[5].Value = M_DATAPROCSRV;
				cmd.Parameters[6].Value = M_WEBSRV;
				cmd.Parameters[7].Value = M_DBSRV;
				cmd.ExecuteNonQuery();
				if (NEWID != null) NEWID = cmd.Parameters[8].Value;
			}
		}
		/// <summary>
		/// Accesses the PC_ADD_PROC_OPERATION procedure in the DB.
		/// </summary>
		public class PC_ADD_PROC_OPERATION
		{
			private PC_ADD_PROC_OPERATION() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL PC_ADD_PROC_OPERATION(:p_1,:p_2,:p_3,:p_4,:p_5,:p_6,:p_7)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.DateTime, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_7", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Output);
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			/// <param name = "O_MACHINEID">the value of O_MACHINEID to be used for the procedure call.</param>
			/// <param name = "O_PROGSETID">the value of O_PROGSETID to be used for the procedure call.</param>
			/// <param name = "O_REQID">the value of O_REQID to be used for the procedure call.</param>
			/// <param name = "O_PARENTOPID">the value of O_PARENTOPID to be used for the procedure call.</param>
			/// <param name = "O_STARTTIME">the value of O_STARTTIME to be used for the procedure call.</param>
			/// <param name = "O_NOTES">the value of O_NOTES to be used for the procedure call.</param>
			/// <param name = "NEWID">the value of NEWID obtained from the procedure call as an output. If null, the output is ignored.</param>
			public static void Call(object O_MACHINEID, object O_PROGSETID, object O_REQID, object O_PARENTOPID, object O_STARTTIME, object O_NOTES, ref object NEWID)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = O_MACHINEID;
				cmd.Parameters[1].Value = O_PROGSETID;
				cmd.Parameters[2].Value = O_REQID;
				cmd.Parameters[3].Value = O_PARENTOPID;
				cmd.Parameters[4].Value = O_STARTTIME;
				cmd.Parameters[5].Value = O_NOTES;
				cmd.ExecuteNonQuery();
				if (NEWID != null) NEWID = cmd.Parameters[6].Value;
			}
		}
		/// <summary>
		/// Accesses the PC_ADD_PROC_OPERATION_BRICK procedure in the DB.
		/// </summary>
		public class PC_ADD_PROC_OPERATION_BRICK
		{
			private PC_ADD_PROC_OPERATION_BRICK() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL PC_ADD_PROC_OPERATION_BRICK(:p_1,:p_2,:p_3,:p_4,:p_5,:p_6,:p_7,:p_8)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.DateTime, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_7", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_8", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Output);
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			/// <param name = "O_MACHINEID">the value of O_MACHINEID to be used for the procedure call.</param>
			/// <param name = "O_PROGSETID">the value of O_PROGSETID to be used for the procedure call.</param>
			/// <param name = "O_REQID">the value of O_REQID to be used for the procedure call.</param>
			/// <param name = "O_BRICKID">the value of O_BRICKID to be used for the procedure call.</param>
			/// <param name = "O_PARENTOPID">the value of O_PARENTOPID to be used for the procedure call.</param>
			/// <param name = "O_STARTTIME">the value of O_STARTTIME to be used for the procedure call.</param>
			/// <param name = "O_NOTES">the value of O_NOTES to be used for the procedure call.</param>
			/// <param name = "NEWID">the value of NEWID obtained from the procedure call as an output. If null, the output is ignored.</param>
			public static void Call(object O_MACHINEID, object O_PROGSETID, object O_REQID, object O_BRICKID, object O_PARENTOPID, object O_STARTTIME, object O_NOTES, ref object NEWID)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = O_MACHINEID;
				cmd.Parameters[1].Value = O_PROGSETID;
				cmd.Parameters[2].Value = O_REQID;
				cmd.Parameters[3].Value = O_BRICKID;
				cmd.Parameters[4].Value = O_PARENTOPID;
				cmd.Parameters[5].Value = O_STARTTIME;
				cmd.Parameters[6].Value = O_NOTES;
				cmd.ExecuteNonQuery();
				if (NEWID != null) NEWID = cmd.Parameters[7].Value;
			}
		}
		/// <summary>
		/// Accesses the PC_ADD_PROC_OPERATION_PLATE procedure in the DB.
		/// </summary>
		public class PC_ADD_PROC_OPERATION_PLATE
		{
			private PC_ADD_PROC_OPERATION_PLATE() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL PC_ADD_PROC_OPERATION_PLATE(:p_1,:p_2,:p_3,:p_4,:p_5,:p_6,:p_7,:p_8,:p_9,:p_10)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_7", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_8", SySal.OperaDb.OperaDbType.DateTime, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_9", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_10", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Output);
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			/// <param name = "O_MACHINEID">the value of O_MACHINEID to be used for the procedure call.</param>
			/// <param name = "O_PROGSETID">the value of O_PROGSETID to be used for the procedure call.</param>
			/// <param name = "O_REQID">the value of O_REQID to be used for the procedure call.</param>
			/// <param name = "O_BRICKID">the value of O_BRICKID to be used for the procedure call.</param>
			/// <param name = "O_PLATEID">the value of O_PLATEID to be used for the procedure call.</param>
			/// <param name = "O_PARENTOPID">the value of O_PARENTOPID to be used for the procedure call.</param>
			/// <param name = "O_CALIBRATION">the value of O_CALIBRATION to be used for the procedure call.</param>
			/// <param name = "O_STARTTIME">the value of O_STARTTIME to be used for the procedure call.</param>
			/// <param name = "O_NOTES">the value of O_NOTES to be used for the procedure call.</param>
			/// <param name = "NEWID">the value of NEWID obtained from the procedure call as an output. If null, the output is ignored.</param>
			public static void Call(object O_MACHINEID, object O_PROGSETID, object O_REQID, object O_BRICKID, object O_PLATEID, object O_PARENTOPID, object O_CALIBRATION, object O_STARTTIME, object O_NOTES, ref object NEWID)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = O_MACHINEID;
				cmd.Parameters[1].Value = O_PROGSETID;
				cmd.Parameters[2].Value = O_REQID;
				cmd.Parameters[3].Value = O_BRICKID;
				cmd.Parameters[4].Value = O_PLATEID;
				cmd.Parameters[5].Value = O_PARENTOPID;
				cmd.Parameters[6].Value = O_CALIBRATION;
				cmd.Parameters[7].Value = O_STARTTIME;
				cmd.Parameters[8].Value = O_NOTES;
				cmd.ExecuteNonQuery();
				if (NEWID != null) NEWID = cmd.Parameters[9].Value;
			}
		}
		/// <summary>
		/// Accesses the PC_ADD_PROGRAMSETTINGS procedure in the DB.
		/// </summary>
		public class PC_ADD_PROGRAMSETTINGS
		{
			private PC_ADD_PROGRAMSETTINGS() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL PC_ADD_PROGRAMSETTINGS(:p_1,:p_2,:p_3,:p_4,:p_5,:p_6,:p_7,:p_8)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_7", SySal.OperaDb.OperaDbType.CLOB, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_8", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Output);
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			/// <param name = "P_DESC">the value of P_DESC to be used for the procedure call.</param>
			/// <param name = "P_EXE">the value of P_EXE to be used for the procedure call.</param>
			/// <param name = "P_AUTHID">the value of P_AUTHID to be used for the procedure call.</param>
			/// <param name = "P_DRIVERLEVEL">the value of P_DRIVERLEVEL to be used for the procedure call.</param>
			/// <param name = "P_USETEMPLATEMARKS">the value of P_USETEMPLATEMARKS to be used for the procedure call.</param>
			/// <param name = "P_MARKSET">the value of P_MARKSET to be used for the procedure call.</param>
			/// <param name = "P_SETTINGS">the value of P_SETTINGS to be used for the procedure call.</param>
			/// <param name = "NEWID">the value of NEWID obtained from the procedure call as an output. If null, the output is ignored.</param>
			public static void Call(object P_DESC, object P_EXE, object P_AUTHID, object P_DRIVERLEVEL, object P_USETEMPLATEMARKS, object P_MARKSET, object P_SETTINGS, ref object NEWID)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = P_DESC;
				cmd.Parameters[1].Value = P_EXE;
				cmd.Parameters[2].Value = P_AUTHID;
				cmd.Parameters[3].Value = P_DRIVERLEVEL;
				cmd.Parameters[4].Value = P_USETEMPLATEMARKS;
				cmd.Parameters[5].Value = P_MARKSET;
				cmd.Parameters[6].Value = P_SETTINGS;
				cmd.ExecuteNonQuery();
				if (NEWID != null) NEWID = cmd.Parameters[7].Value;
			}
		}
		/// <summary>
		/// Accesses the PC_ADD_PUBLISHER procedure in the DB.
		/// </summary>
		public class PC_ADD_PUBLISHER
		{
			private PC_ADD_PUBLISHER() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL PC_ADD_PUBLISHER(:p_1,:p_2,:p_3,:p_4,:p_5)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			/// <param name = "NAME">the value of NAME to be used for the procedure call.</param>
			/// <param name = "SHORTNAME">the value of SHORTNAME to be used for the procedure call.</param>
			/// <param name = "PUB_TYPE">the value of PUB_TYPE to be used for the procedure call.</param>
			/// <param name = "RANGEMIN">the value of RANGEMIN to be used for the procedure call.</param>
			/// <param name = "RANGEMAX">the value of RANGEMAX to be used for the procedure call.</param>
			public static void Call(object NAME, object SHORTNAME, object PUB_TYPE, object RANGEMIN, object RANGEMAX)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = NAME;
				cmd.Parameters[1].Value = SHORTNAME;
				cmd.Parameters[2].Value = PUB_TYPE;
				cmd.Parameters[3].Value = RANGEMIN;
				cmd.Parameters[4].Value = RANGEMAX;
				cmd.ExecuteNonQuery();
			}
		}
		/// <summary>
		/// Accesses the PC_ADD_SITE procedure in the DB.
		/// </summary>
		public class PC_ADD_SITE
		{
			private PC_ADD_SITE() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL PC_ADD_SITE(:p_1,:p_2,:p_3,:p_4,:p_5)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Output);
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			/// <param name = "S_NAME">the value of S_NAME to be used for the procedure call.</param>
			/// <param name = "S_LATITUDE">the value of S_LATITUDE to be used for the procedure call.</param>
			/// <param name = "S_LONGITUDE">the value of S_LONGITUDE to be used for the procedure call.</param>
			/// <param name = "S_LOCALTIMEFUSE">the value of S_LOCALTIMEFUSE to be used for the procedure call.</param>
			/// <param name = "NEWID">the value of NEWID obtained from the procedure call as an output. If null, the output is ignored.</param>
			public static void Call(object S_NAME, object S_LATITUDE, object S_LONGITUDE, object S_LOCALTIMEFUSE, ref object NEWID)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = S_NAME;
				cmd.Parameters[1].Value = S_LATITUDE;
				cmd.Parameters[2].Value = S_LONGITUDE;
				cmd.Parameters[3].Value = S_LOCALTIMEFUSE;
				cmd.ExecuteNonQuery();
				if (NEWID != null) NEWID = cmd.Parameters[4].Value;
			}
		}
		/// <summary>
		/// Accesses the PC_ADD_USER procedure in the DB.
		/// </summary>
		public class PC_ADD_USER
		{
			private PC_ADD_USER() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL PC_ADD_USER(:p_1,:p_2,:p_3,:p_4,:p_5,:p_6,:p_7,:p_8,:p_9,:p_10)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_7", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_8", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_9", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_10", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Output);
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			/// <param name = "S_ID">the value of S_ID to be used for the procedure call.</param>
			/// <param name = "U_USERNAME">the value of U_USERNAME to be used for the procedure call.</param>
			/// <param name = "U_PWD">the value of U_PWD to be used for the procedure call.</param>
			/// <param name = "U_NAME">the value of U_NAME to be used for the procedure call.</param>
			/// <param name = "U_SURNAME">the value of U_SURNAME to be used for the procedure call.</param>
			/// <param name = "U_INST">the value of U_INST to be used for the procedure call.</param>
			/// <param name = "U_EMAIL">the value of U_EMAIL to be used for the procedure call.</param>
			/// <param name = "U_ADDRESS">the value of U_ADDRESS to be used for the procedure call.</param>
			/// <param name = "U_PHONE">the value of U_PHONE to be used for the procedure call.</param>
			/// <param name = "NEWID">the value of NEWID obtained from the procedure call as an output. If null, the output is ignored.</param>
			public static void Call(object S_ID, object U_USERNAME, object U_PWD, object U_NAME, object U_SURNAME, object U_INST, object U_EMAIL, object U_ADDRESS, object U_PHONE, ref object NEWID)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = S_ID;
				cmd.Parameters[1].Value = U_USERNAME;
				cmd.Parameters[2].Value = U_PWD;
				cmd.Parameters[3].Value = U_NAME;
				cmd.Parameters[4].Value = U_SURNAME;
				cmd.Parameters[5].Value = U_INST;
				cmd.Parameters[6].Value = U_EMAIL;
				cmd.Parameters[7].Value = U_ADDRESS;
				cmd.Parameters[8].Value = U_PHONE;
				cmd.ExecuteNonQuery();
				if (NEWID != null) NEWID = cmd.Parameters[9].Value;
			}
		}
		/// <summary>
		/// Accesses the PC_AUTOEXTEND_TABLESPACES procedure in the DB.
		/// </summary>
		public class PC_AUTOEXTEND_TABLESPACES
		{
			private PC_AUTOEXTEND_TABLESPACES() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL PC_AUTOEXTEND_TABLESPACES()");
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			public static void Call()
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.ExecuteNonQuery();
			}
		}
		/// <summary>
		/// Accesses the PC_CALIBRATE_PLATE procedure in the DB.
		/// </summary>
		public class PC_CALIBRATE_PLATE
		{
			private PC_CALIBRATE_PLATE() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL PC_CALIBRATE_PLATE(:p_1,:p_2,:p_3,:p_4,:p_5,:p_6,:p_7,:p_8,:p_9,:p_10,:p_11)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_7", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_8", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_9", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_10", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_11", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			/// <param name = "B_BRICKID">the value of B_BRICKID to be used for the procedure call.</param>
			/// <param name = "P_PLATEID">the value of P_PLATEID to be used for the procedure call.</param>
			/// <param name = "O_PROCOP">the value of O_PROCOP to be used for the procedure call.</param>
			/// <param name = "O_MARKSETS">the value of O_MARKSETS to be used for the procedure call.</param>
			/// <param name = "C_Z">the value of C_Z to be used for the procedure call.</param>
			/// <param name = "C_MAPXX">the value of C_MAPXX to be used for the procedure call.</param>
			/// <param name = "C_MAPXY">the value of C_MAPXY to be used for the procedure call.</param>
			/// <param name = "C_MAPYX">the value of C_MAPYX to be used for the procedure call.</param>
			/// <param name = "C_MAPYY">the value of C_MAPYY to be used for the procedure call.</param>
			/// <param name = "C_MAPDX">the value of C_MAPDX to be used for the procedure call.</param>
			/// <param name = "C_MAPDY">the value of C_MAPDY to be used for the procedure call.</param>
			public static void Call(object B_BRICKID, object P_PLATEID, object O_PROCOP, object O_MARKSETS, object C_Z, object C_MAPXX, object C_MAPXY, object C_MAPYX, object C_MAPYY, object C_MAPDX, object C_MAPDY)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = B_BRICKID;
				cmd.Parameters[1].Value = P_PLATEID;
				cmd.Parameters[2].Value = O_PROCOP;
				cmd.Parameters[3].Value = O_MARKSETS;
				cmd.Parameters[4].Value = C_Z;
				cmd.Parameters[5].Value = C_MAPXX;
				cmd.Parameters[6].Value = C_MAPXY;
				cmd.Parameters[7].Value = C_MAPYX;
				cmd.Parameters[8].Value = C_MAPYY;
				cmd.Parameters[9].Value = C_MAPDX;
				cmd.Parameters[10].Value = C_MAPDY;
				cmd.ExecuteNonQuery();
			}
		}
		/// <summary>
		/// Accesses the PC_CHECK_LOGIN procedure in the DB.
		/// </summary>
		public class PC_CHECK_LOGIN
		{
			private PC_CHECK_LOGIN() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL PC_CHECK_LOGIN(:p_1,:p_2,:p_3)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Output);
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			/// <param name = "U_USERNAME">the value of U_USERNAME to be used for the procedure call.</param>
			/// <param name = "U_PWD">the value of U_PWD to be used for the procedure call.</param>
			/// <param name = "U_ID">the value of U_ID obtained from the procedure call as an output. If null, the output is ignored.</param>
			public static void Call(object U_USERNAME, object U_PWD, ref object U_ID)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = U_USERNAME;
				cmd.Parameters[1].Value = U_PWD;
				cmd.ExecuteNonQuery();
				if (U_ID != null) U_ID = cmd.Parameters[2].Value;
			}
		}
		/// <summary>
		/// Accesses the PC_CS_AUTO_ADD_SPACE procedure in the DB.
		/// </summary>
		public class PC_CS_AUTO_ADD_SPACE
		{
			private PC_CS_AUTO_ADD_SPACE() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL PC_CS_AUTO_ADD_SPACE(:p_1)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			/// <param name = "I_ID_CS_EVENTBRICK">the value of I_ID_CS_EVENTBRICK to be used for the procedure call.</param>
			public static void Call(object I_ID_CS_EVENTBRICK)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = I_ID_CS_EVENTBRICK;
				cmd.ExecuteNonQuery();
			}
		}
		/// <summary>
		/// Accesses the PC_DEL_MACHINE procedure in the DB.
		/// </summary>
		public class PC_DEL_MACHINE
		{
			private PC_DEL_MACHINE() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL PC_DEL_MACHINE(:p_1)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			/// <param name = "M_ID">the value of M_ID to be used for the procedure call.</param>
			public static void Call(object M_ID)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = M_ID;
				cmd.ExecuteNonQuery();
			}
		}
		/// <summary>
		/// Accesses the PC_DEL_PRIVILEGES procedure in the DB.
		/// </summary>
		public class PC_DEL_PRIVILEGES
		{
			private PC_DEL_PRIVILEGES() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL PC_DEL_PRIVILEGES(:p_1,:p_2)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			/// <param name = "U_ID">the value of U_ID to be used for the procedure call.</param>
			/// <param name = "S_ID">the value of S_ID to be used for the procedure call.</param>
			public static void Call(object U_ID, object S_ID)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = U_ID;
				cmd.Parameters[1].Value = S_ID;
				cmd.ExecuteNonQuery();
			}
		}
		/// <summary>
		/// Accesses the PC_DEL_PROGRAMSETTINGS procedure in the DB.
		/// </summary>
		public class PC_DEL_PROGRAMSETTINGS
		{
			private PC_DEL_PROGRAMSETTINGS() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL PC_DEL_PROGRAMSETTINGS(:p_1)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			/// <param name = "P_ID">the value of P_ID to be used for the procedure call.</param>
			public static void Call(object P_ID)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = P_ID;
				cmd.ExecuteNonQuery();
			}
		}
		/// <summary>
		/// Accesses the PC_DEL_SITE procedure in the DB.
		/// </summary>
		public class PC_DEL_SITE
		{
			private PC_DEL_SITE() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL PC_DEL_SITE(:p_1)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			/// <param name = "S_ID">the value of S_ID to be used for the procedure call.</param>
			public static void Call(object S_ID)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = S_ID;
				cmd.ExecuteNonQuery();
			}
		}
		/// <summary>
		/// Accesses the PC_DEL_USER procedure in the DB.
		/// </summary>
		public class PC_DEL_USER
		{
			private PC_DEL_USER() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL PC_DEL_USER(:p_1)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			/// <param name = "U_ID">the value of U_ID to be used for the procedure call.</param>
			public static void Call(object U_ID)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = U_ID;
				cmd.ExecuteNonQuery();
			}
		}
		/// <summary>
		/// Accesses the PC_DISABLE_TABLE_LOCK procedure in the DB.
		/// </summary>
		public class PC_DISABLE_TABLE_LOCK
		{
			private PC_DISABLE_TABLE_LOCK() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL PC_DISABLE_TABLE_LOCK()");
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			public static void Call()
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.ExecuteNonQuery();
			}
		}
		/// <summary>
		/// Accesses the PC_EMPTY_VOLUMESLICES procedure in the DB.
		/// </summary>
		public class PC_EMPTY_VOLUMESLICES
		{
			private PC_EMPTY_VOLUMESLICES() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL PC_EMPTY_VOLUMESLICES(:p_1,:p_2,:p_3)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			/// <param name = "ID_BRICK">the value of ID_BRICK to be used for the procedure call.</param>
			/// <param name = "ID_PROCOP">the value of ID_PROCOP to be used for the procedure call.</param>
			/// <param name = "ID_PL">the value of ID_PL to be used for the procedure call.</param>
			public static void Call(object ID_BRICK, object ID_PROCOP, object ID_PL)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = ID_BRICK;
				cmd.Parameters[1].Value = ID_PROCOP;
				cmd.Parameters[2].Value = ID_PL;
				cmd.ExecuteNonQuery();
			}
		}
		/// <summary>
		/// Accesses the PC_ENABLE_TABLE_LOCK procedure in the DB.
		/// </summary>
		public class PC_ENABLE_TABLE_LOCK
		{
			private PC_ENABLE_TABLE_LOCK() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL PC_ENABLE_TABLE_LOCK()");
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			public static void Call()
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.ExecuteNonQuery();
			}
		}
		/// <summary>
		/// Accesses the PC_FAIL_OPERATION procedure in the DB.
		/// </summary>
		public class PC_FAIL_OPERATION
		{
			private PC_FAIL_OPERATION() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL PC_FAIL_OPERATION(:p_1,:p_2)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.DateTime, System.Data.ParameterDirection.Input);
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			/// <param name = "O_ID">the value of O_ID to be used for the procedure call.</param>
			/// <param name = "O_FINISHTIME">the value of O_FINISHTIME to be used for the procedure call.</param>
			public static void Call(object O_ID, object O_FINISHTIME)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = O_ID;
				cmd.Parameters[1].Value = O_FINISHTIME;
				cmd.ExecuteNonQuery();
			}
		}
		/// <summary>
		/// Accesses the PC_FIND_BRICKSET procedure in the DB.
		/// </summary>
		public class PC_FIND_BRICKSET
		{
			private PC_FIND_BRICKSET() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL PC_FIND_BRICKSET(:p_1,:p_2)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Output);
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			/// <param name = "I_ID">the value of I_ID to be used for the procedure call.</param>
			/// <param name = "O_SET">the value of O_SET obtained from the procedure call as an output. If null, the output is ignored.</param>
			public static void Call(object I_ID, ref object O_SET)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = I_ID;
				cmd.ExecuteNonQuery();
				if (O_SET != null) O_SET = cmd.Parameters[1].Value;
			}
		}
		/// <summary>
		/// Accesses the PC_GET_PRIVILEGES procedure in the DB.
		/// </summary>
		public class PC_GET_PRIVILEGES
		{
			private PC_GET_PRIVILEGES() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL PC_GET_PRIVILEGES(:p_1,:p_2,:p_3,:p_4,:p_5,:p_6,:p_7,:p_8,:p_9)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Output);
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Output);
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Output);
				newcmd.Parameters.Add("p_7", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Output);
				newcmd.Parameters.Add("p_8", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Output);
				newcmd.Parameters.Add("p_9", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Output);
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			/// <param name = "U_USERID">the value of U_USERID to be used for the procedure call.</param>
			/// <param name = "S_SITEID">the value of S_SITEID to be used for the procedure call.</param>
			/// <param name = "PWD">the value of PWD to be used for the procedure call.</param>
			/// <param name = "P_SCAN">the value of P_SCAN obtained from the procedure call as an output. If null, the output is ignored.</param>
			/// <param name = "P_WEBAN">the value of P_WEBAN obtained from the procedure call as an output. If null, the output is ignored.</param>
			/// <param name = "P_DATAPROC">the value of P_DATAPROC obtained from the procedure call as an output. If null, the output is ignored.</param>
			/// <param name = "P_DATADWNL">the value of P_DATADWNL obtained from the procedure call as an output. If null, the output is ignored.</param>
			/// <param name = "P_PROCSTART">the value of P_PROCSTART obtained from the procedure call as an output. If null, the output is ignored.</param>
			/// <param name = "P_ADMIN">the value of P_ADMIN obtained from the procedure call as an output. If null, the output is ignored.</param>
			public static void Call(object U_USERID, object S_SITEID, object PWD, ref object P_SCAN, ref object P_WEBAN, ref object P_DATAPROC, ref object P_DATADWNL, ref object P_PROCSTART, ref object P_ADMIN)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = U_USERID;
				cmd.Parameters[1].Value = S_SITEID;
				cmd.Parameters[2].Value = PWD;
				cmd.ExecuteNonQuery();
				if (P_SCAN != null) P_SCAN = cmd.Parameters[3].Value;
				if (P_WEBAN != null) P_WEBAN = cmd.Parameters[4].Value;
				if (P_DATAPROC != null) P_DATAPROC = cmd.Parameters[5].Value;
				if (P_DATADWNL != null) P_DATADWNL = cmd.Parameters[6].Value;
				if (P_PROCSTART != null) P_PROCSTART = cmd.Parameters[7].Value;
				if (P_ADMIN != null) P_ADMIN = cmd.Parameters[8].Value;
			}
		}
		/// <summary>
		/// Accesses the PC_GET_PRIVILEGES_ADM procedure in the DB.
		/// </summary>
		public class PC_GET_PRIVILEGES_ADM
		{
			private PC_GET_PRIVILEGES_ADM() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL PC_GET_PRIVILEGES_ADM(:p_1,:p_2,:p_3,:p_4,:p_5,:p_6,:p_7,:p_8,:p_9,:p_10)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Output);
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Output);
				newcmd.Parameters.Add("p_7", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Output);
				newcmd.Parameters.Add("p_8", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Output);
				newcmd.Parameters.Add("p_9", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Output);
				newcmd.Parameters.Add("p_10", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Output);
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			/// <param name = "ADM_ID">the value of ADM_ID to be used for the procedure call.</param>
			/// <param name = "ADM_PWD">the value of ADM_PWD to be used for the procedure call.</param>
			/// <param name = "U_USERID">the value of U_USERID to be used for the procedure call.</param>
			/// <param name = "S_SITEID">the value of S_SITEID to be used for the procedure call.</param>
			/// <param name = "P_SCAN">the value of P_SCAN obtained from the procedure call as an output. If null, the output is ignored.</param>
			/// <param name = "P_WEBAN">the value of P_WEBAN obtained from the procedure call as an output. If null, the output is ignored.</param>
			/// <param name = "P_DATAPROC">the value of P_DATAPROC obtained from the procedure call as an output. If null, the output is ignored.</param>
			/// <param name = "P_DATADWNL">the value of P_DATADWNL obtained from the procedure call as an output. If null, the output is ignored.</param>
			/// <param name = "P_PROCSTART">the value of P_PROCSTART obtained from the procedure call as an output. If null, the output is ignored.</param>
			/// <param name = "P_ADMIN">the value of P_ADMIN obtained from the procedure call as an output. If null, the output is ignored.</param>
			public static void Call(object ADM_ID, object ADM_PWD, object U_USERID, object S_SITEID, ref object P_SCAN, ref object P_WEBAN, ref object P_DATAPROC, ref object P_DATADWNL, ref object P_PROCSTART, ref object P_ADMIN)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = ADM_ID;
				cmd.Parameters[1].Value = ADM_PWD;
				cmd.Parameters[2].Value = U_USERID;
				cmd.Parameters[3].Value = S_SITEID;
				cmd.ExecuteNonQuery();
				if (P_SCAN != null) P_SCAN = cmd.Parameters[4].Value;
				if (P_WEBAN != null) P_WEBAN = cmd.Parameters[5].Value;
				if (P_DATAPROC != null) P_DATAPROC = cmd.Parameters[6].Value;
				if (P_DATADWNL != null) P_DATADWNL = cmd.Parameters[7].Value;
				if (P_PROCSTART != null) P_PROCSTART = cmd.Parameters[8].Value;
				if (P_ADMIN != null) P_ADMIN = cmd.Parameters[9].Value;
			}
		}
		/// <summary>
		/// Accesses the PC_GET_PWD procedure in the DB.
		/// </summary>
		public class PC_GET_PWD
		{
			private PC_GET_PWD() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL PC_GET_PWD(:p_1,:p_2,:p_3,:p_4)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Output);
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			/// <param name = "ADMIN_ID">the value of ADMIN_ID to be used for the procedure call.</param>
			/// <param name = "ADMIN_PWD">the value of ADMIN_PWD to be used for the procedure call.</param>
			/// <param name = "USER_ID">the value of USER_ID to be used for the procedure call.</param>
			/// <param name = "USER_PWD">the value of USER_PWD obtained from the procedure call as an output. If null, the output is ignored.</param>
			public static void Call(object ADMIN_ID, object ADMIN_PWD, object USER_ID, ref object USER_PWD)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = ADMIN_ID;
				cmd.Parameters[1].Value = ADMIN_PWD;
				cmd.Parameters[2].Value = USER_ID;
				cmd.ExecuteNonQuery();
				if (USER_PWD != null) USER_PWD = cmd.Parameters[3].Value;
			}
		}
		/// <summary>
		/// Accesses the PC_JOB_SLEEP procedure in the DB.
		/// </summary>
		public class PC_JOB_SLEEP
		{
			private PC_JOB_SLEEP() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL PC_JOB_SLEEP(:p_1)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			/// <param name = "SECS">the value of SECS to be used for the procedure call.</param>
			public static void Call(object SECS)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = SECS;
				cmd.ExecuteNonQuery();
			}
		}
		/// <summary>
		/// Accesses the PC_REFRESH_PUBLISHER procedure in the DB.
		/// </summary>
		public class PC_REFRESH_PUBLISHER
		{
			private PC_REFRESH_PUBLISHER() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL PC_REFRESH_PUBLISHER(:p_1,:p_2)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			/// <param name = "I_NAME">the value of I_NAME to be used for the procedure call.</param>
			/// <param name = "SHORTNAME">the value of SHORTNAME to be used for the procedure call.</param>
			public static void Call(object I_NAME, object SHORTNAME)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = I_NAME;
				cmd.Parameters[1].Value = SHORTNAME;
				cmd.ExecuteNonQuery();
			}
		}
		/// <summary>
		/// Accesses the PC_REMOVE_BRICK_SET procedure in the DB.
		/// </summary>
		public class PC_REMOVE_BRICK_SET
		{
			private PC_REMOVE_BRICK_SET() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL PC_REMOVE_BRICK_SET(:p_1)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			/// <param name = "IDSET">the value of IDSET to be used for the procedure call.</param>
			public static void Call(object IDSET)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = IDSET;
				cmd.ExecuteNonQuery();
			}
		}
		/// <summary>
		/// Accesses the PC_REMOVE_BRICK_SPACE procedure in the DB.
		/// </summary>
		public class PC_REMOVE_BRICK_SPACE
		{
			private PC_REMOVE_BRICK_SPACE() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL PC_REMOVE_BRICK_SPACE(:p_1,:p_2)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			/// <param name = "OLDID">the value of OLDID to be used for the procedure call.</param>
			/// <param name = "IDSET">the value of IDSET to be used for the procedure call.</param>
			public static void Call(object OLDID, object IDSET)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = OLDID;
				cmd.Parameters[1].Value = IDSET;
				cmd.ExecuteNonQuery();
			}
		}
		/// <summary>
		/// Accesses the PC_REMOVE_CS_OR_BRICK procedure in the DB.
		/// </summary>
		public class PC_REMOVE_CS_OR_BRICK
		{
			private PC_REMOVE_CS_OR_BRICK() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL PC_REMOVE_CS_OR_BRICK(:p_1,:p_2)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			/// <param name = "B_IDBRICK">the value of B_IDBRICK to be used for the procedure call.</param>
			/// <param name = "B_SET">the value of B_SET to be used for the procedure call.</param>
			public static void Call(object B_IDBRICK, object B_SET)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = B_IDBRICK;
				cmd.Parameters[1].Value = B_SET;
				cmd.ExecuteNonQuery();
			}
		}
		/// <summary>
		/// Accesses the PC_REMOVE_PUBLISHER procedure in the DB.
		/// </summary>
		public class PC_REMOVE_PUBLISHER
		{
			private PC_REMOVE_PUBLISHER() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL PC_REMOVE_PUBLISHER(:p_1)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			/// <param name = "NAME">the value of NAME to be used for the procedure call.</param>
			public static void Call(object NAME)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = NAME;
				cmd.ExecuteNonQuery();
			}
		}
		/// <summary>
		/// Accesses the PC_RESET_PLATE_CALIBRATION procedure in the DB.
		/// </summary>
		public class PC_RESET_PLATE_CALIBRATION
		{
			private PC_RESET_PLATE_CALIBRATION() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL PC_RESET_PLATE_CALIBRATION(:p_1,:p_2)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			/// <param name = "ID_BRICK">the value of ID_BRICK to be used for the procedure call.</param>
			/// <param name = "ID_PLATE">the value of ID_PLATE to be used for the procedure call.</param>
			public static void Call(object ID_BRICK, object ID_PLATE)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = ID_BRICK;
				cmd.Parameters[1].Value = ID_PLATE;
				cmd.ExecuteNonQuery();
			}
		}
		/// <summary>
		/// Accesses the PC_SCANBACK_CANCEL_PATH procedure in the DB.
		/// </summary>
		public class PC_SCANBACK_CANCEL_PATH
		{
			private PC_SCANBACK_CANCEL_PATH() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL PC_SCANBACK_CANCEL_PATH(:p_1,:p_2,:p_3)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			/// <param name = "P_BRICKID">the value of P_BRICKID to be used for the procedure call.</param>
			/// <param name = "P_PATHID">the value of P_PATHID to be used for the procedure call.</param>
			/// <param name = "P_PLATEID">the value of P_PLATEID to be used for the procedure call.</param>
			public static void Call(object P_BRICKID, object P_PATHID, object P_PLATEID)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = P_BRICKID;
				cmd.Parameters[1].Value = P_PATHID;
				cmd.Parameters[2].Value = P_PLATEID;
				cmd.ExecuteNonQuery();
			}
		}
		/// <summary>
		/// Accesses the PC_SCANBACK_CANDIDATE procedure in the DB.
		/// </summary>
		public class PC_SCANBACK_CANDIDATE
		{
			private PC_SCANBACK_CANDIDATE() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL PC_SCANBACK_CANDIDATE(:p_1,:p_2,:p_3,:p_4,:p_5,:p_6)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			/// <param name = "P_BRICKID">the value of P_BRICKID to be used for the procedure call.</param>
			/// <param name = "P_PLATEID">the value of P_PLATEID to be used for the procedure call.</param>
			/// <param name = "P_PATHID">the value of P_PATHID to be used for the procedure call.</param>
			/// <param name = "P_ZONEID">the value of P_ZONEID to be used for the procedure call.</param>
			/// <param name = "P_CANDID">the value of P_CANDID to be used for the procedure call.</param>
			/// <param name = "P_MANUAL">the value of P_MANUAL to be used for the procedure call.</param>
			public static void Call(object P_BRICKID, object P_PLATEID, object P_PATHID, object P_ZONEID, object P_CANDID, object P_MANUAL)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = P_BRICKID;
				cmd.Parameters[1].Value = P_PLATEID;
				cmd.Parameters[2].Value = P_PATHID;
				cmd.Parameters[3].Value = P_ZONEID;
				cmd.Parameters[4].Value = P_CANDID;
				cmd.Parameters[5].Value = P_MANUAL;
				cmd.ExecuteNonQuery();
			}
		}
		/// <summary>
		/// Accesses the PC_SCANBACK_DAMAGEDZONE procedure in the DB.
		/// </summary>
		public class PC_SCANBACK_DAMAGEDZONE
		{
			private PC_SCANBACK_DAMAGEDZONE() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL PC_SCANBACK_DAMAGEDZONE(:p_1,:p_2,:p_3,:p_4)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			/// <param name = "P_BRICKID">the value of P_BRICKID to be used for the procedure call.</param>
			/// <param name = "P_PLATEID">the value of P_PLATEID to be used for the procedure call.</param>
			/// <param name = "P_PATHID">the value of P_PATHID to be used for the procedure call.</param>
			/// <param name = "P_DAMAGE">the value of P_DAMAGE to be used for the procedure call.</param>
			public static void Call(object P_BRICKID, object P_PLATEID, object P_PATHID, object P_DAMAGE)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = P_BRICKID;
				cmd.Parameters[1].Value = P_PLATEID;
				cmd.Parameters[2].Value = P_PATHID;
				cmd.Parameters[3].Value = P_DAMAGE;
				cmd.ExecuteNonQuery();
			}
		}
		/// <summary>
		/// Accesses the PC_SCANBACK_DELETE_PREDICTIONS procedure in the DB.
		/// </summary>
		public class PC_SCANBACK_DELETE_PREDICTIONS
		{
			private PC_SCANBACK_DELETE_PREDICTIONS() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL PC_SCANBACK_DELETE_PREDICTIONS(:p_1,:p_2,:p_3)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			/// <param name = "ID_BRICK">the value of ID_BRICK to be used for the procedure call.</param>
			/// <param name = "ID_PROCOP">the value of ID_PROCOP to be used for the procedure call.</param>
			/// <param name = "ID_PL">the value of ID_PL to be used for the procedure call.</param>
			public static void Call(object ID_BRICK, object ID_PROCOP, object ID_PL)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = ID_BRICK;
				cmd.Parameters[1].Value = ID_PROCOP;
				cmd.Parameters[2].Value = ID_PL;
				cmd.ExecuteNonQuery();
			}
		}
		/// <summary>
		/// Accesses the PC_SCANBACK_FORK procedure in the DB.
		/// </summary>
		public class PC_SCANBACK_FORK
		{
			private PC_SCANBACK_FORK() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL PC_SCANBACK_FORK(:p_1,:p_2,:p_3,:p_4,:p_5,:p_6)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			/// <param name = "P_BRICKID">the value of P_BRICKID to be used for the procedure call.</param>
			/// <param name = "P_PLATEID">the value of P_PLATEID to be used for the procedure call.</param>
			/// <param name = "P_PATHID">the value of P_PATHID to be used for the procedure call.</param>
			/// <param name = "P_ZONEID">the value of P_ZONEID to be used for the procedure call.</param>
			/// <param name = "P_CANDID">the value of P_CANDID to be used for the procedure call.</param>
			/// <param name = "P_FORKID">the value of P_FORKID to be used for the procedure call.</param>
			public static void Call(object P_BRICKID, object P_PLATEID, object P_PATHID, object P_ZONEID, object P_CANDID, object P_FORKID)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = P_BRICKID;
				cmd.Parameters[1].Value = P_PLATEID;
				cmd.Parameters[2].Value = P_PATHID;
				cmd.Parameters[3].Value = P_ZONEID;
				cmd.Parameters[4].Value = P_CANDID;
				cmd.Parameters[5].Value = P_FORKID;
				cmd.ExecuteNonQuery();
			}
		}
		/// <summary>
		/// Accesses the PC_SCANBACK_NOCANDIDATE procedure in the DB.
		/// </summary>
		public class PC_SCANBACK_NOCANDIDATE
		{
			private PC_SCANBACK_NOCANDIDATE() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL PC_SCANBACK_NOCANDIDATE(:p_1,:p_2,:p_3,:p_4)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			/// <param name = "P_BRICKID">the value of P_BRICKID to be used for the procedure call.</param>
			/// <param name = "P_PLATEID">the value of P_PLATEID to be used for the procedure call.</param>
			/// <param name = "P_PATHID">the value of P_PATHID to be used for the procedure call.</param>
			/// <param name = "P_ZONEID">the value of P_ZONEID to be used for the procedure call.</param>
			public static void Call(object P_BRICKID, object P_PLATEID, object P_PATHID, object P_ZONEID)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = P_BRICKID;
				cmd.Parameters[1].Value = P_PLATEID;
				cmd.Parameters[2].Value = P_PATHID;
				cmd.Parameters[3].Value = P_ZONEID;
				cmd.ExecuteNonQuery();
			}
		}
		/// <summary>
		/// Accesses the PC_SET_CSCAND_SBPATH procedure in the DB.
		/// </summary>
		public class PC_SET_CSCAND_SBPATH
		{
			private PC_SET_CSCAND_SBPATH() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL PC_SET_CSCAND_SBPATH(:p_1,:p_2,:p_3,:p_4,:p_5)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			/// <param name = "C_BRICKID">the value of C_BRICKID to be used for the procedure call.</param>
			/// <param name = "C_CAND">the value of C_CAND to be used for the procedure call.</param>
			/// <param name = "S_BRICKID">the value of S_BRICKID to be used for the procedure call.</param>
			/// <param name = "S_PROCOPID">the value of S_PROCOPID to be used for the procedure call.</param>
			/// <param name = "S_PATH">the value of S_PATH to be used for the procedure call.</param>
			public static void Call(object C_BRICKID, object C_CAND, object S_BRICKID, object S_PROCOPID, object S_PATH)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = C_BRICKID;
				cmd.Parameters[1].Value = C_CAND;
				cmd.Parameters[2].Value = S_BRICKID;
				cmd.Parameters[3].Value = S_PROCOPID;
				cmd.Parameters[4].Value = S_PATH;
				cmd.ExecuteNonQuery();
			}
		}
		/// <summary>
		/// Accesses the PC_SET_MACHINE procedure in the DB.
		/// </summary>
		public class PC_SET_MACHINE
		{
			private PC_SET_MACHINE() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL PC_SET_MACHINE(:p_1,:p_2,:p_3,:p_4,:p_5,:p_6,:p_7,:p_8)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_7", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_8", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			/// <param name = "M_ID">the value of M_ID to be used for the procedure call.</param>
			/// <param name = "M_NAME">the value of M_NAME to be used for the procedure call.</param>
			/// <param name = "M_ADDRESS">the value of M_ADDRESS to be used for the procedure call.</param>
			/// <param name = "M_SCANSRV">the value of M_SCANSRV to be used for the procedure call.</param>
			/// <param name = "M_BATCHSRV">the value of M_BATCHSRV to be used for the procedure call.</param>
			/// <param name = "M_DATAPROCSRV">the value of M_DATAPROCSRV to be used for the procedure call.</param>
			/// <param name = "M_WEBSRV">the value of M_WEBSRV to be used for the procedure call.</param>
			/// <param name = "M_DBSRV">the value of M_DBSRV to be used for the procedure call.</param>
			public static void Call(object M_ID, object M_NAME, object M_ADDRESS, object M_SCANSRV, object M_BATCHSRV, object M_DATAPROCSRV, object M_WEBSRV, object M_DBSRV)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = M_ID;
				cmd.Parameters[1].Value = M_NAME;
				cmd.Parameters[2].Value = M_ADDRESS;
				cmd.Parameters[3].Value = M_SCANSRV;
				cmd.Parameters[4].Value = M_BATCHSRV;
				cmd.Parameters[5].Value = M_DATAPROCSRV;
				cmd.Parameters[6].Value = M_WEBSRV;
				cmd.Parameters[7].Value = M_DBSRV;
				cmd.ExecuteNonQuery();
			}
		}
		/// <summary>
		/// Accesses the PC_SET_PASSWORD procedure in the DB.
		/// </summary>
		public class PC_SET_PASSWORD
		{
			private PC_SET_PASSWORD() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL PC_SET_PASSWORD(:p_1,:p_2,:p_3)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			/// <param name = "U_ID">the value of U_ID to be used for the procedure call.</param>
			/// <param name = "OLDPWD">the value of OLDPWD to be used for the procedure call.</param>
			/// <param name = "NEWPWD">the value of NEWPWD to be used for the procedure call.</param>
			public static void Call(object U_ID, object OLDPWD, object NEWPWD)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = U_ID;
				cmd.Parameters[1].Value = OLDPWD;
				cmd.Parameters[2].Value = NEWPWD;
				cmd.ExecuteNonQuery();
			}
		}
		/// <summary>
		/// Accesses the PC_SET_PLATE_DAMAGED procedure in the DB.
		/// </summary>
		public class PC_SET_PLATE_DAMAGED
		{
			private PC_SET_PLATE_DAMAGED() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL PC_SET_PLATE_DAMAGED(:p_1,:p_2,:p_3,:p_4)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			/// <param name = "ID_BRICK">the value of ID_BRICK to be used for the procedure call.</param>
			/// <param name = "ID_PL">the value of ID_PL to be used for the procedure call.</param>
			/// <param name = "ID_PROCOP">the value of ID_PROCOP to be used for the procedure call.</param>
			/// <param name = "DAMAGECODE">the value of DAMAGECODE to be used for the procedure call.</param>
			public static void Call(object ID_BRICK, object ID_PL, object ID_PROCOP, object DAMAGECODE)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = ID_BRICK;
				cmd.Parameters[1].Value = ID_PL;
				cmd.Parameters[2].Value = ID_PROCOP;
				cmd.Parameters[3].Value = DAMAGECODE;
				cmd.ExecuteNonQuery();
			}
		}
		/// <summary>
		/// Accesses the PC_SET_PLATE_Z procedure in the DB.
		/// </summary>
		public class PC_SET_PLATE_Z
		{
			private PC_SET_PLATE_Z() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL PC_SET_PLATE_Z(:p_1,:p_2,:p_3)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			/// <param name = "B_BRICK">the value of B_BRICK to be used for the procedure call.</param>
			/// <param name = "B_IDPL">the value of B_IDPL to be used for the procedure call.</param>
			/// <param name = "B_PLZ">the value of B_PLZ to be used for the procedure call.</param>
			public static void Call(object B_BRICK, object B_IDPL, object B_PLZ)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = B_BRICK;
				cmd.Parameters[1].Value = B_IDPL;
				cmd.Parameters[2].Value = B_PLZ;
				cmd.ExecuteNonQuery();
			}
		}
		/// <summary>
		/// Accesses the PC_SET_PREDTRACKS_SBPATH procedure in the DB.
		/// </summary>
		public class PC_SET_PREDTRACKS_SBPATH
		{
			private PC_SET_PREDTRACKS_SBPATH() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL PC_SET_PREDTRACKS_SBPATH(:p_1,:p_2,:p_3,:p_4,:p_5)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			/// <param name = "S_EVENT">the value of S_EVENT to be used for the procedure call.</param>
			/// <param name = "S_TRACK">the value of S_TRACK to be used for the procedure call.</param>
			/// <param name = "S_BRICKID">the value of S_BRICKID to be used for the procedure call.</param>
			/// <param name = "S_PROCOPID">the value of S_PROCOPID to be used for the procedure call.</param>
			/// <param name = "S_PATH">the value of S_PATH to be used for the procedure call.</param>
			public static void Call(object S_EVENT, object S_TRACK, object S_BRICKID, object S_PROCOPID, object S_PATH)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = S_EVENT;
				cmd.Parameters[1].Value = S_TRACK;
				cmd.Parameters[2].Value = S_BRICKID;
				cmd.Parameters[3].Value = S_PROCOPID;
				cmd.Parameters[4].Value = S_PATH;
				cmd.ExecuteNonQuery();
			}
		}
		/// <summary>
		/// Accesses the PC_SET_PREDTRACK_SBPATH procedure in the DB.
		/// </summary>
		public class PC_SET_PREDTRACK_SBPATH
		{
			private PC_SET_PREDTRACK_SBPATH() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL PC_SET_PREDTRACK_SBPATH(:p_1,:p_2,:p_3,:p_4,:p_5)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			/// <param name = "S_EVENT">the value of S_EVENT to be used for the procedure call.</param>
			/// <param name = "S_TRACK">the value of S_TRACK to be used for the procedure call.</param>
			/// <param name = "S_BRICKID">the value of S_BRICKID to be used for the procedure call.</param>
			/// <param name = "S_PROCOPID">the value of S_PROCOPID to be used for the procedure call.</param>
			/// <param name = "S_PATH">the value of S_PATH to be used for the procedure call.</param>
			public static void Call(object S_EVENT, object S_TRACK, object S_BRICKID, object S_PROCOPID, object S_PATH)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = S_EVENT;
				cmd.Parameters[1].Value = S_TRACK;
				cmd.Parameters[2].Value = S_BRICKID;
				cmd.Parameters[3].Value = S_PROCOPID;
				cmd.Parameters[4].Value = S_PATH;
				cmd.ExecuteNonQuery();
			}
		}
		/// <summary>
		/// Accesses the PC_SET_PRIVILEGES procedure in the DB.
		/// </summary>
		public class PC_SET_PRIVILEGES
		{
			private PC_SET_PRIVILEGES() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL PC_SET_PRIVILEGES(:p_1,:p_2,:p_3,:p_4,:p_5,:p_6,:p_7,:p_8)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_7", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_8", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			/// <param name = "U_ID">the value of U_ID to be used for the procedure call.</param>
			/// <param name = "S_ID">the value of S_ID to be used for the procedure call.</param>
			/// <param name = "P_SCAN">the value of P_SCAN to be used for the procedure call.</param>
			/// <param name = "P_WEBAN">the value of P_WEBAN to be used for the procedure call.</param>
			/// <param name = "P_DATAPROC">the value of P_DATAPROC to be used for the procedure call.</param>
			/// <param name = "P_DATADWNL">the value of P_DATADWNL to be used for the procedure call.</param>
			/// <param name = "P_PROCSTART">the value of P_PROCSTART to be used for the procedure call.</param>
			/// <param name = "P_ADMIN">the value of P_ADMIN to be used for the procedure call.</param>
			public static void Call(object U_ID, object S_ID, object P_SCAN, object P_WEBAN, object P_DATAPROC, object P_DATADWNL, object P_PROCSTART, object P_ADMIN)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = U_ID;
				cmd.Parameters[1].Value = S_ID;
				cmd.Parameters[2].Value = P_SCAN;
				cmd.Parameters[3].Value = P_WEBAN;
				cmd.Parameters[4].Value = P_DATAPROC;
				cmd.Parameters[5].Value = P_DATADWNL;
				cmd.Parameters[6].Value = P_PROCSTART;
				cmd.Parameters[7].Value = P_ADMIN;
				cmd.ExecuteNonQuery();
			}
		}
		/// <summary>
		/// Accesses the PC_SET_SBPATH_VOLUME procedure in the DB.
		/// </summary>
		public class PC_SET_SBPATH_VOLUME
		{
			private PC_SET_SBPATH_VOLUME() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL PC_SET_SBPATH_VOLUME(:p_1,:p_2,:p_3,:p_4,:p_5,:p_6)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			/// <param name = "S_BRICKID">the value of S_BRICKID to be used for the procedure call.</param>
			/// <param name = "S_PROCOPID">the value of S_PROCOPID to be used for the procedure call.</param>
			/// <param name = "S_PATH">the value of S_PATH to be used for the procedure call.</param>
			/// <param name = "S_PLATE">the value of S_PLATE to be used for the procedure call.</param>
			/// <param name = "V_PROCOPID">the value of V_PROCOPID to be used for the procedure call.</param>
			/// <param name = "V_VOLUME">the value of V_VOLUME to be used for the procedure call.</param>
			public static void Call(object S_BRICKID, object S_PROCOPID, object S_PATH, object S_PLATE, object V_PROCOPID, object V_VOLUME)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = S_BRICKID;
				cmd.Parameters[1].Value = S_PROCOPID;
				cmd.Parameters[2].Value = S_PATH;
				cmd.Parameters[3].Value = S_PLATE;
				cmd.Parameters[4].Value = V_PROCOPID;
				cmd.Parameters[5].Value = V_VOLUME;
				cmd.ExecuteNonQuery();
			}
		}
		/// <summary>
		/// Accesses the PC_SET_SITE procedure in the DB.
		/// </summary>
		public class PC_SET_SITE
		{
			private PC_SET_SITE() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL PC_SET_SITE(:p_1,:p_2,:p_3,:p_4,:p_5)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Double, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			/// <param name = "S_ID">the value of S_ID to be used for the procedure call.</param>
			/// <param name = "S_NAME">the value of S_NAME to be used for the procedure call.</param>
			/// <param name = "S_LATITUDE">the value of S_LATITUDE to be used for the procedure call.</param>
			/// <param name = "S_LONGITUDE">the value of S_LONGITUDE to be used for the procedure call.</param>
			/// <param name = "S_LOCALTIMEFUSE">the value of S_LOCALTIMEFUSE to be used for the procedure call.</param>
			public static void Call(object S_ID, object S_NAME, object S_LATITUDE, object S_LONGITUDE, object S_LOCALTIMEFUSE)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = S_ID;
				cmd.Parameters[1].Value = S_NAME;
				cmd.Parameters[2].Value = S_LATITUDE;
				cmd.Parameters[3].Value = S_LONGITUDE;
				cmd.Parameters[4].Value = S_LOCALTIMEFUSE;
				cmd.ExecuteNonQuery();
			}
		}
		/// <summary>
		/// Accesses the PC_SET_USER procedure in the DB.
		/// </summary>
		public class PC_SET_USER
		{
			private PC_SET_USER() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL PC_SET_USER(:p_1,:p_2,:p_3,:p_4,:p_5,:p_6,:p_7,:p_8,:p_9)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_6", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_7", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_8", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_9", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			/// <param name = "U_ID">the value of U_ID to be used for the procedure call.</param>
			/// <param name = "U_USERNAME">the value of U_USERNAME to be used for the procedure call.</param>
			/// <param name = "U_PWD">the value of U_PWD to be used for the procedure call.</param>
			/// <param name = "U_NAME">the value of U_NAME to be used for the procedure call.</param>
			/// <param name = "U_SURNAME">the value of U_SURNAME to be used for the procedure call.</param>
			/// <param name = "U_INST">the value of U_INST to be used for the procedure call.</param>
			/// <param name = "U_EMAIL">the value of U_EMAIL to be used for the procedure call.</param>
			/// <param name = "U_ADDRESS">the value of U_ADDRESS to be used for the procedure call.</param>
			/// <param name = "U_PHONE">the value of U_PHONE to be used for the procedure call.</param>
			public static void Call(object U_ID, object U_USERNAME, object U_PWD, object U_NAME, object U_SURNAME, object U_INST, object U_EMAIL, object U_ADDRESS, object U_PHONE)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = U_ID;
				cmd.Parameters[1].Value = U_USERNAME;
				cmd.Parameters[2].Value = U_PWD;
				cmd.Parameters[3].Value = U_NAME;
				cmd.Parameters[4].Value = U_SURNAME;
				cmd.Parameters[5].Value = U_INST;
				cmd.Parameters[6].Value = U_EMAIL;
				cmd.Parameters[7].Value = U_ADDRESS;
				cmd.Parameters[8].Value = U_PHONE;
				cmd.ExecuteNonQuery();
			}
		}
		/// <summary>
		/// Accesses the PC_SET_VOLUMESLICE_ZONE procedure in the DB.
		/// </summary>
		public class PC_SET_VOLUMESLICE_ZONE
		{
			private PC_SET_VOLUMESLICE_ZONE() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL PC_SET_VOLUMESLICE_ZONE(:p_1,:p_2,:p_3,:p_4,:p_5)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_3", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_4", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_5", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			/// <param name = "P_BRICKID">the value of P_BRICKID to be used for the procedure call.</param>
			/// <param name = "P_PLATEID">the value of P_PLATEID to be used for the procedure call.</param>
			/// <param name = "P_VOLID">the value of P_VOLID to be used for the procedure call.</param>
			/// <param name = "P_ZONEID">the value of P_ZONEID to be used for the procedure call.</param>
			/// <param name = "P_DAMAGED">the value of P_DAMAGED to be used for the procedure call.</param>
			public static void Call(object P_BRICKID, object P_PLATEID, object P_VOLID, object P_ZONEID, object P_DAMAGED)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = P_BRICKID;
				cmd.Parameters[1].Value = P_PLATEID;
				cmd.Parameters[2].Value = P_VOLID;
				cmd.Parameters[3].Value = P_ZONEID;
				cmd.Parameters[4].Value = P_DAMAGED;
				cmd.ExecuteNonQuery();
			}
		}
		/// <summary>
		/// Accesses the PC_SUCCESS_OPERATION procedure in the DB.
		/// </summary>
		public class PC_SUCCESS_OPERATION
		{
			private PC_SUCCESS_OPERATION() {}
			internal static bool Prepared = false;
			internal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();
			private static SySal.OperaDb.OperaDbCommand InitCommand()
			{
				SySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand("CALL PC_SUCCESS_OPERATION(:p_1,:p_2)");
				newcmd.Parameters.Add("p_1", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
				newcmd.Parameters.Add("p_2", SySal.OperaDb.OperaDbType.DateTime, System.Data.ParameterDirection.Input);
				return newcmd;
			}
			/// <summary>
			/// Calls the procedure.
			/// </summary>
			/// <param name = "O_ID">the value of O_ID to be used for the procedure call.</param>
			/// <param name = "O_FINISHTIME">the value of O_FINISHTIME to be used for the procedure call.</param>
			public static void Call(object O_ID, object O_FINISHTIME)
			{
				if (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };
				cmd.Parameters[0].Value = O_ID;
				cmd.Parameters[1].Value = O_FINISHTIME;
				cmd.ExecuteNonQuery();
			}
		}
		static internal SySal.OperaDb.OperaDbConnection m_DB = null;
		/// <summary>
		/// The DB connection currently used by the Schema class. Must be set before using any child class.
		/// </summary>
		static public SySal.OperaDb.OperaDbConnection DB
		{
			get { return m_DB; }
			set
			{
				if (LZ_BUFFERFLUSH.Prepared) { LZ_BUFFERFLUSH.cmd.Connection = null; LZ_BUFFERFLUSH.Prepared = false; }
				if (LZ_GRAINS.Prepared) { LZ_GRAINS.cmd.Connection = null; LZ_GRAINS.Prepared = false; }
				if (LZ_MACHINEVARS.Prepared) { LZ_MACHINEVARS.cmd.Connection = null; LZ_MACHINEVARS.Prepared = false; }
				if (LZ_MIPBASETRACKS.Prepared) { LZ_MIPBASETRACKS.cmd.Connection = null; LZ_MIPBASETRACKS.Prepared = false; }
				if (LZ_MIPMICROTRACKS.Prepared) { LZ_MIPMICROTRACKS.cmd.Connection = null; LZ_MIPMICROTRACKS.Prepared = false; }
				if (LZ_PATTERN_MATCH.Prepared) { LZ_PATTERN_MATCH.cmd.Connection = null; LZ_PATTERN_MATCH.Prepared = false; }
				if (LZ_PUBLISHERS.Prepared) { LZ_PUBLISHERS.cmd.Connection = null; LZ_PUBLISHERS.Prepared = false; }
				if (LZ_SCANBACK_CANCEL_PATH.Prepared) { LZ_SCANBACK_CANCEL_PATH.cmd.Connection = null; LZ_SCANBACK_CANCEL_PATH.Prepared = false; }
				if (LZ_SCANBACK_CANDIDATE.Prepared) { LZ_SCANBACK_CANDIDATE.cmd.Connection = null; LZ_SCANBACK_CANDIDATE.Prepared = false; }
				if (LZ_SCANBACK_DAMAGEDZONE.Prepared) { LZ_SCANBACK_DAMAGEDZONE.cmd.Connection = null; LZ_SCANBACK_DAMAGEDZONE.Prepared = false; }
				if (LZ_SCANBACK_FORK.Prepared) { LZ_SCANBACK_FORK.cmd.Connection = null; LZ_SCANBACK_FORK.Prepared = false; }
				if (LZ_SCANBACK_NOCANDIDATE.Prepared) { LZ_SCANBACK_NOCANDIDATE.cmd.Connection = null; LZ_SCANBACK_NOCANDIDATE.Prepared = false; }
				if (LZ_SET_VOLUMESLICE_ZONE.Prepared) { LZ_SET_VOLUMESLICE_ZONE.cmd.Connection = null; LZ_SET_VOLUMESLICE_ZONE.Prepared = false; }
				if (LZ_SITEVARS.Prepared) { LZ_SITEVARS.cmd.Connection = null; LZ_SITEVARS.Prepared = false; }
				if (LZ_TOKENS.Prepared) { LZ_TOKENS.cmd.Connection = null; LZ_TOKENS.Prepared = false; }
				if (LZ_VIEWS.Prepared) { LZ_VIEWS.cmd.Connection = null; LZ_VIEWS.Prepared = false; }
				if (LZ_ZONES.Prepared) { LZ_ZONES.cmd.Connection = null; LZ_ZONES.Prepared = false; }
				if (TB_ALIGNED_MIPMICROTRACKS.Prepared) { TB_ALIGNED_MIPMICROTRACKS.cmd.Connection = null; TB_ALIGNED_MIPMICROTRACKS.Prepared = false; }
				if (TB_ALIGNED_SIDES.Prepared) { TB_ALIGNED_SIDES.cmd.Connection = null; TB_ALIGNED_SIDES.Prepared = false; }
				if (TB_ALIGNED_SLICES.Prepared) { TB_ALIGNED_SLICES.cmd.Connection = null; TB_ALIGNED_SLICES.Prepared = false; }
				if (TB_BRICK_SETS.Prepared) { TB_BRICK_SETS.cmd.Connection = null; TB_BRICK_SETS.Prepared = false; }
				if (TB_B_CSCANDS_SBPATHS.Prepared) { TB_B_CSCANDS_SBPATHS.cmd.Connection = null; TB_B_CSCANDS_SBPATHS.Prepared = false; }
				if (TB_B_PREDTRACKS_CSCANDS.Prepared) { TB_B_PREDTRACKS_CSCANDS.cmd.Connection = null; TB_B_PREDTRACKS_CSCANDS.Prepared = false; }
				if (TB_B_PREDTRACKS_SBPATHS.Prepared) { TB_B_PREDTRACKS_SBPATHS.cmd.Connection = null; TB_B_PREDTRACKS_SBPATHS.Prepared = false; }
				if (TB_B_SBPATHS_SBPATHS.Prepared) { TB_B_SBPATHS_SBPATHS.cmd.Connection = null; TB_B_SBPATHS_SBPATHS.Prepared = false; }
				if (TB_B_SBPATHS_VOLUMES.Prepared) { TB_B_SBPATHS_VOLUMES.cmd.Connection = null; TB_B_SBPATHS_VOLUMES.Prepared = false; }
				if (TB_B_VOLTKS_SBPATHS.Prepared) { TB_B_VOLTKS_SBPATHS.cmd.Connection = null; TB_B_VOLTKS_SBPATHS.Prepared = false; }
				if (TB_CS_CANDIDATES.Prepared) { TB_CS_CANDIDATES.cmd.Connection = null; TB_CS_CANDIDATES.Prepared = false; }
				if (TB_CS_CANDIDATE_CHECKS.Prepared) { TB_CS_CANDIDATE_CHECKS.cmd.Connection = null; TB_CS_CANDIDATE_CHECKS.Prepared = false; }
				if (TB_CS_CANDIDATE_TRACKS.Prepared) { TB_CS_CANDIDATE_TRACKS.cmd.Connection = null; TB_CS_CANDIDATE_TRACKS.Prepared = false; }
				if (TB_EVENTBRICKS.Prepared) { TB_EVENTBRICKS.cmd.Connection = null; TB_EVENTBRICKS.Prepared = false; }
				if (TB_GRAINS.Prepared) { TB_GRAINS.cmd.Connection = null; TB_GRAINS.Prepared = false; }
				if (TB_MACHINES.Prepared) { TB_MACHINES.cmd.Connection = null; TB_MACHINES.Prepared = false; }
				if (TB_MIPBASETRACKS.Prepared) { TB_MIPBASETRACKS.cmd.Connection = null; TB_MIPBASETRACKS.Prepared = false; }
				if (TB_MIPMICROTRACKS.Prepared) { TB_MIPMICROTRACKS.cmd.Connection = null; TB_MIPMICROTRACKS.Prepared = false; }
				if (TB_PACKAGED_SW.Prepared) { TB_PACKAGED_SW.cmd.Connection = null; TB_PACKAGED_SW.Prepared = false; }
				if (TB_PATTERN_MATCH.Prepared) { TB_PATTERN_MATCH.cmd.Connection = null; TB_PATTERN_MATCH.Prepared = false; }
				if (TB_PEANUT_BRICKALIGN.Prepared) { TB_PEANUT_BRICKALIGN.cmd.Connection = null; TB_PEANUT_BRICKALIGN.Prepared = false; }
				if (TB_PEANUT_BRICKINFO.Prepared) { TB_PEANUT_BRICKINFO.cmd.Connection = null; TB_PEANUT_BRICKINFO.Prepared = false; }
				if (TB_PEANUT_HITS.Prepared) { TB_PEANUT_HITS.cmd.Connection = null; TB_PEANUT_HITS.Prepared = false; }
				if (TB_PEANUT_PREDTRACKBRICKS.Prepared) { TB_PEANUT_PREDTRACKBRICKS.cmd.Connection = null; TB_PEANUT_PREDTRACKBRICKS.Prepared = false; }
				if (TB_PEANUT_PREDTRACKS.Prepared) { TB_PEANUT_PREDTRACKS.cmd.Connection = null; TB_PEANUT_PREDTRACKS.Prepared = false; }
				if (TB_PEANUT_TRACKHITLINKS.Prepared) { TB_PEANUT_TRACKHITLINKS.cmd.Connection = null; TB_PEANUT_TRACKHITLINKS.Prepared = false; }
				if (TB_PLATES.Prepared) { TB_PLATES.cmd.Connection = null; TB_PLATES.Prepared = false; }
				if (TB_PLATE_CALIBRATIONS.Prepared) { TB_PLATE_CALIBRATIONS.cmd.Connection = null; TB_PLATE_CALIBRATIONS.Prepared = false; }
				if (TB_PLATE_DAMAGENOTICES.Prepared) { TB_PLATE_DAMAGENOTICES.cmd.Connection = null; TB_PLATE_DAMAGENOTICES.Prepared = false; }
				if (TB_PREDICTED_BRICKS.Prepared) { TB_PREDICTED_BRICKS.cmd.Connection = null; TB_PREDICTED_BRICKS.Prepared = false; }
				if (TB_PREDICTED_EVENTS.Prepared) { TB_PREDICTED_EVENTS.cmd.Connection = null; TB_PREDICTED_EVENTS.Prepared = false; }
				if (TB_PREDICTED_TRACKS.Prepared) { TB_PREDICTED_TRACKS.cmd.Connection = null; TB_PREDICTED_TRACKS.Prepared = false; }
				if (TB_PRIVILEGES.Prepared) { TB_PRIVILEGES.cmd.Connection = null; TB_PRIVILEGES.Prepared = false; }
				if (TB_PROC_OPERATIONS.Prepared) { TB_PROC_OPERATIONS.cmd.Connection = null; TB_PROC_OPERATIONS.Prepared = false; }
				if (TB_PROGRAMSETTINGS.Prepared) { TB_PROGRAMSETTINGS.cmd.Connection = null; TB_PROGRAMSETTINGS.Prepared = false; }
				if (TB_RECONSTRUCTIONS.Prepared) { TB_RECONSTRUCTIONS.cmd.Connection = null; TB_RECONSTRUCTIONS.Prepared = false; }
				if (TB_RECONSTRUCTION_LISTS.Prepared) { TB_RECONSTRUCTION_LISTS.cmd.Connection = null; TB_RECONSTRUCTION_LISTS.Prepared = false; }
				if (TB_RECONSTRUCTION_OPTIONS.Prepared) { TB_RECONSTRUCTION_OPTIONS.cmd.Connection = null; TB_RECONSTRUCTION_OPTIONS.Prepared = false; }
				if (TB_RECONSTRUCTION_SETS.Prepared) { TB_RECONSTRUCTION_SETS.cmd.Connection = null; TB_RECONSTRUCTION_SETS.Prepared = false; }
				if (TB_SCANBACK_CHECKRESULTS.Prepared) { TB_SCANBACK_CHECKRESULTS.cmd.Connection = null; TB_SCANBACK_CHECKRESULTS.Prepared = false; }
				if (TB_SCANBACK_PATHS.Prepared) { TB_SCANBACK_PATHS.cmd.Connection = null; TB_SCANBACK_PATHS.Prepared = false; }
				if (TB_SCANBACK_PREDICTIONS.Prepared) { TB_SCANBACK_PREDICTIONS.cmd.Connection = null; TB_SCANBACK_PREDICTIONS.Prepared = false; }
				if (TB_SITES.Prepared) { TB_SITES.cmd.Connection = null; TB_SITES.Prepared = false; }
				if (TB_TEMPLATEMARKSETS.Prepared) { TB_TEMPLATEMARKSETS.cmd.Connection = null; TB_TEMPLATEMARKSETS.Prepared = false; }
				if (TB_USERS.Prepared) { TB_USERS.cmd.Connection = null; TB_USERS.Prepared = false; }
				if (TB_VERTICES.Prepared) { TB_VERTICES.cmd.Connection = null; TB_VERTICES.Prepared = false; }
				if (TB_VERTICES_ATTR.Prepared) { TB_VERTICES_ATTR.cmd.Connection = null; TB_VERTICES_ATTR.Prepared = false; }
				if (TB_VERTICES_FIT.Prepared) { TB_VERTICES_FIT.cmd.Connection = null; TB_VERTICES_FIT.Prepared = false; }
				if (TB_VIEWS.Prepared) { TB_VIEWS.cmd.Connection = null; TB_VIEWS.Prepared = false; }
				if (TB_VOLTKS_ALIGNMUTKS.Prepared) { TB_VOLTKS_ALIGNMUTKS.cmd.Connection = null; TB_VOLTKS_ALIGNMUTKS.Prepared = false; }
				if (TB_VOLUMES.Prepared) { TB_VOLUMES.cmd.Connection = null; TB_VOLUMES.Prepared = false; }
				if (TB_VOLUMETRACKS.Prepared) { TB_VOLUMETRACKS.cmd.Connection = null; TB_VOLUMETRACKS.Prepared = false; }
				if (TB_VOLUMETRACKS_ATTR.Prepared) { TB_VOLUMETRACKS_ATTR.cmd.Connection = null; TB_VOLUMETRACKS_ATTR.Prepared = false; }
				if (TB_VOLUMETRACKS_FIT.Prepared) { TB_VOLUMETRACKS_FIT.cmd.Connection = null; TB_VOLUMETRACKS_FIT.Prepared = false; }
				if (TB_VOLUME_SLICES.Prepared) { TB_VOLUME_SLICES.cmd.Connection = null; TB_VOLUME_SLICES.Prepared = false; }
				if (TB_ZONES.Prepared) { TB_ZONES.cmd.Connection = null; TB_ZONES.Prepared = false; }
				if (LP_ADD_PROC_OPERATION.Prepared) { LP_ADD_PROC_OPERATION.cmd.Connection = null; LP_ADD_PROC_OPERATION.Prepared = false; }
				if (LP_ADD_PROC_OPERATION_BRICK.Prepared) { LP_ADD_PROC_OPERATION_BRICK.cmd.Connection = null; LP_ADD_PROC_OPERATION_BRICK.Prepared = false; }
				if (LP_ADD_PROC_OPERATION_PLATE.Prepared) { LP_ADD_PROC_OPERATION_PLATE.cmd.Connection = null; LP_ADD_PROC_OPERATION_PLATE.Prepared = false; }
				if (LP_CHECK_ACCESS.Prepared) { LP_CHECK_ACCESS.cmd.Connection = null; LP_CHECK_ACCESS.Prepared = false; }
				if (LP_CHECK_TOKEN_OWNERSHIP.Prepared) { LP_CHECK_TOKEN_OWNERSHIP.cmd.Connection = null; LP_CHECK_TOKEN_OWNERSHIP.Prepared = false; }
				if (LP_CLEAN_ORPHAN_TOKENS.Prepared) { LP_CLEAN_ORPHAN_TOKENS.cmd.Connection = null; LP_CLEAN_ORPHAN_TOKENS.Prepared = false; }
				if (LP_DATABUFFER_FLUSH.Prepared) { LP_DATABUFFER_FLUSH.cmd.Connection = null; LP_DATABUFFER_FLUSH.Prepared = false; }
				if (LP_FAIL_OPERATION.Prepared) { LP_FAIL_OPERATION.cmd.Connection = null; LP_FAIL_OPERATION.Prepared = false; }
				if (LP_SUCCESS_OPERATION.Prepared) { LP_SUCCESS_OPERATION.cmd.Connection = null; LP_SUCCESS_OPERATION.Prepared = false; }
				if (PC_ADD_BRICK.Prepared) { PC_ADD_BRICK.cmd.Connection = null; PC_ADD_BRICK.Prepared = false; }
				if (PC_ADD_BRICK_EASY.Prepared) { PC_ADD_BRICK_EASY.cmd.Connection = null; PC_ADD_BRICK_EASY.Prepared = false; }
				if (PC_ADD_BRICK_EASY_Z.Prepared) { PC_ADD_BRICK_EASY_Z.cmd.Connection = null; PC_ADD_BRICK_EASY_Z.Prepared = false; }
				if (PC_ADD_BRICK_SET.Prepared) { PC_ADD_BRICK_SET.cmd.Connection = null; PC_ADD_BRICK_SET.Prepared = false; }
				if (PC_ADD_BRICK_SPACE.Prepared) { PC_ADD_BRICK_SPACE.cmd.Connection = null; PC_ADD_BRICK_SPACE.Prepared = false; }
				if (PC_ADD_CS_DOUBLET.Prepared) { PC_ADD_CS_DOUBLET.cmd.Connection = null; PC_ADD_CS_DOUBLET.Prepared = false; }
				if (PC_ADD_MACHINE.Prepared) { PC_ADD_MACHINE.cmd.Connection = null; PC_ADD_MACHINE.Prepared = false; }
				if (PC_ADD_PROC_OPERATION.Prepared) { PC_ADD_PROC_OPERATION.cmd.Connection = null; PC_ADD_PROC_OPERATION.Prepared = false; }
				if (PC_ADD_PROC_OPERATION_BRICK.Prepared) { PC_ADD_PROC_OPERATION_BRICK.cmd.Connection = null; PC_ADD_PROC_OPERATION_BRICK.Prepared = false; }
				if (PC_ADD_PROC_OPERATION_PLATE.Prepared) { PC_ADD_PROC_OPERATION_PLATE.cmd.Connection = null; PC_ADD_PROC_OPERATION_PLATE.Prepared = false; }
				if (PC_ADD_PROGRAMSETTINGS.Prepared) { PC_ADD_PROGRAMSETTINGS.cmd.Connection = null; PC_ADD_PROGRAMSETTINGS.Prepared = false; }
				if (PC_ADD_PUBLISHER.Prepared) { PC_ADD_PUBLISHER.cmd.Connection = null; PC_ADD_PUBLISHER.Prepared = false; }
				if (PC_ADD_SITE.Prepared) { PC_ADD_SITE.cmd.Connection = null; PC_ADD_SITE.Prepared = false; }
				if (PC_ADD_USER.Prepared) { PC_ADD_USER.cmd.Connection = null; PC_ADD_USER.Prepared = false; }
				if (PC_AUTOEXTEND_TABLESPACES.Prepared) { PC_AUTOEXTEND_TABLESPACES.cmd.Connection = null; PC_AUTOEXTEND_TABLESPACES.Prepared = false; }
				if (PC_CALIBRATE_PLATE.Prepared) { PC_CALIBRATE_PLATE.cmd.Connection = null; PC_CALIBRATE_PLATE.Prepared = false; }
				if (PC_CHECK_LOGIN.Prepared) { PC_CHECK_LOGIN.cmd.Connection = null; PC_CHECK_LOGIN.Prepared = false; }
				if (PC_CS_AUTO_ADD_SPACE.Prepared) { PC_CS_AUTO_ADD_SPACE.cmd.Connection = null; PC_CS_AUTO_ADD_SPACE.Prepared = false; }
				if (PC_DEL_MACHINE.Prepared) { PC_DEL_MACHINE.cmd.Connection = null; PC_DEL_MACHINE.Prepared = false; }
				if (PC_DEL_PRIVILEGES.Prepared) { PC_DEL_PRIVILEGES.cmd.Connection = null; PC_DEL_PRIVILEGES.Prepared = false; }
				if (PC_DEL_PROGRAMSETTINGS.Prepared) { PC_DEL_PROGRAMSETTINGS.cmd.Connection = null; PC_DEL_PROGRAMSETTINGS.Prepared = false; }
				if (PC_DEL_SITE.Prepared) { PC_DEL_SITE.cmd.Connection = null; PC_DEL_SITE.Prepared = false; }
				if (PC_DEL_USER.Prepared) { PC_DEL_USER.cmd.Connection = null; PC_DEL_USER.Prepared = false; }
				if (PC_DISABLE_TABLE_LOCK.Prepared) { PC_DISABLE_TABLE_LOCK.cmd.Connection = null; PC_DISABLE_TABLE_LOCK.Prepared = false; }
				if (PC_EMPTY_VOLUMESLICES.Prepared) { PC_EMPTY_VOLUMESLICES.cmd.Connection = null; PC_EMPTY_VOLUMESLICES.Prepared = false; }
				if (PC_ENABLE_TABLE_LOCK.Prepared) { PC_ENABLE_TABLE_LOCK.cmd.Connection = null; PC_ENABLE_TABLE_LOCK.Prepared = false; }
				if (PC_FAIL_OPERATION.Prepared) { PC_FAIL_OPERATION.cmd.Connection = null; PC_FAIL_OPERATION.Prepared = false; }
				if (PC_FIND_BRICKSET.Prepared) { PC_FIND_BRICKSET.cmd.Connection = null; PC_FIND_BRICKSET.Prepared = false; }
				if (PC_GET_PRIVILEGES.Prepared) { PC_GET_PRIVILEGES.cmd.Connection = null; PC_GET_PRIVILEGES.Prepared = false; }
				if (PC_GET_PRIVILEGES_ADM.Prepared) { PC_GET_PRIVILEGES_ADM.cmd.Connection = null; PC_GET_PRIVILEGES_ADM.Prepared = false; }
				if (PC_GET_PWD.Prepared) { PC_GET_PWD.cmd.Connection = null; PC_GET_PWD.Prepared = false; }
				if (PC_JOB_SLEEP.Prepared) { PC_JOB_SLEEP.cmd.Connection = null; PC_JOB_SLEEP.Prepared = false; }
				if (PC_REFRESH_PUBLISHER.Prepared) { PC_REFRESH_PUBLISHER.cmd.Connection = null; PC_REFRESH_PUBLISHER.Prepared = false; }
				if (PC_REMOVE_BRICK_SET.Prepared) { PC_REMOVE_BRICK_SET.cmd.Connection = null; PC_REMOVE_BRICK_SET.Prepared = false; }
				if (PC_REMOVE_BRICK_SPACE.Prepared) { PC_REMOVE_BRICK_SPACE.cmd.Connection = null; PC_REMOVE_BRICK_SPACE.Prepared = false; }
				if (PC_REMOVE_CS_OR_BRICK.Prepared) { PC_REMOVE_CS_OR_BRICK.cmd.Connection = null; PC_REMOVE_CS_OR_BRICK.Prepared = false; }
				if (PC_REMOVE_PUBLISHER.Prepared) { PC_REMOVE_PUBLISHER.cmd.Connection = null; PC_REMOVE_PUBLISHER.Prepared = false; }
				if (PC_RESET_PLATE_CALIBRATION.Prepared) { PC_RESET_PLATE_CALIBRATION.cmd.Connection = null; PC_RESET_PLATE_CALIBRATION.Prepared = false; }
				if (PC_SCANBACK_CANCEL_PATH.Prepared) { PC_SCANBACK_CANCEL_PATH.cmd.Connection = null; PC_SCANBACK_CANCEL_PATH.Prepared = false; }
				if (PC_SCANBACK_CANDIDATE.Prepared) { PC_SCANBACK_CANDIDATE.cmd.Connection = null; PC_SCANBACK_CANDIDATE.Prepared = false; }
				if (PC_SCANBACK_DAMAGEDZONE.Prepared) { PC_SCANBACK_DAMAGEDZONE.cmd.Connection = null; PC_SCANBACK_DAMAGEDZONE.Prepared = false; }
				if (PC_SCANBACK_DELETE_PREDICTIONS.Prepared) { PC_SCANBACK_DELETE_PREDICTIONS.cmd.Connection = null; PC_SCANBACK_DELETE_PREDICTIONS.Prepared = false; }
				if (PC_SCANBACK_FORK.Prepared) { PC_SCANBACK_FORK.cmd.Connection = null; PC_SCANBACK_FORK.Prepared = false; }
				if (PC_SCANBACK_NOCANDIDATE.Prepared) { PC_SCANBACK_NOCANDIDATE.cmd.Connection = null; PC_SCANBACK_NOCANDIDATE.Prepared = false; }
				if (PC_SET_CSCAND_SBPATH.Prepared) { PC_SET_CSCAND_SBPATH.cmd.Connection = null; PC_SET_CSCAND_SBPATH.Prepared = false; }
				if (PC_SET_MACHINE.Prepared) { PC_SET_MACHINE.cmd.Connection = null; PC_SET_MACHINE.Prepared = false; }
				if (PC_SET_PASSWORD.Prepared) { PC_SET_PASSWORD.cmd.Connection = null; PC_SET_PASSWORD.Prepared = false; }
				if (PC_SET_PLATE_DAMAGED.Prepared) { PC_SET_PLATE_DAMAGED.cmd.Connection = null; PC_SET_PLATE_DAMAGED.Prepared = false; }
				if (PC_SET_PLATE_Z.Prepared) { PC_SET_PLATE_Z.cmd.Connection = null; PC_SET_PLATE_Z.Prepared = false; }
				if (PC_SET_PREDTRACKS_SBPATH.Prepared) { PC_SET_PREDTRACKS_SBPATH.cmd.Connection = null; PC_SET_PREDTRACKS_SBPATH.Prepared = false; }
				if (PC_SET_PREDTRACK_SBPATH.Prepared) { PC_SET_PREDTRACK_SBPATH.cmd.Connection = null; PC_SET_PREDTRACK_SBPATH.Prepared = false; }
				if (PC_SET_PRIVILEGES.Prepared) { PC_SET_PRIVILEGES.cmd.Connection = null; PC_SET_PRIVILEGES.Prepared = false; }
				if (PC_SET_SBPATH_VOLUME.Prepared) { PC_SET_SBPATH_VOLUME.cmd.Connection = null; PC_SET_SBPATH_VOLUME.Prepared = false; }
				if (PC_SET_SITE.Prepared) { PC_SET_SITE.cmd.Connection = null; PC_SET_SITE.Prepared = false; }
				if (PC_SET_USER.Prepared) { PC_SET_USER.cmd.Connection = null; PC_SET_USER.Prepared = false; }
				if (PC_SET_VOLUMESLICE_ZONE.Prepared) { PC_SET_VOLUMESLICE_ZONE.cmd.Connection = null; PC_SET_VOLUMESLICE_ZONE.Prepared = false; }
				if (PC_SUCCESS_OPERATION.Prepared) { PC_SUCCESS_OPERATION.cmd.Connection = null; PC_SUCCESS_OPERATION.Prepared = false; }
				m_DB = value;
			}
		}
	}
}
