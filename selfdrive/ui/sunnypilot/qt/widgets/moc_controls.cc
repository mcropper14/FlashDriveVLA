/****************************************************************************
** Meta object code from reading C++ file 'controls.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.12.8)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "controls.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'controls.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.12.8. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_ElidedLabelSP_t {
    QByteArrayData data[3];
    char stringdata0[23];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_ElidedLabelSP_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_ElidedLabelSP_t qt_meta_stringdata_ElidedLabelSP = {
    {
QT_MOC_LITERAL(0, 0, 13), // "ElidedLabelSP"
QT_MOC_LITERAL(1, 14, 7), // "clicked"
QT_MOC_LITERAL(2, 22, 0) // ""

    },
    "ElidedLabelSP\0clicked\0"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_ElidedLabelSP[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       1,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    0,   19,    2, 0x06 /* Public */,

 // signals: parameters
    QMetaType::Void,

       0        // eod
};

void ElidedLabelSP::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<ElidedLabelSP *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->clicked(); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (ElidedLabelSP::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&ElidedLabelSP::clicked)) {
                *result = 0;
                return;
            }
        }
    }
    Q_UNUSED(_a);
}

QT_INIT_METAOBJECT const QMetaObject ElidedLabelSP::staticMetaObject = { {
    &QLabel::staticMetaObject,
    qt_meta_stringdata_ElidedLabelSP.data,
    qt_meta_data_ElidedLabelSP,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *ElidedLabelSP::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *ElidedLabelSP::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_ElidedLabelSP.stringdata0))
        return static_cast<void*>(this);
    return QLabel::qt_metacast(_clname);
}

int ElidedLabelSP::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QLabel::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 1)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 1;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 1)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 1;
    }
    return _id;
}

// SIGNAL 0
void ElidedLabelSP::clicked()
{
    QMetaObject::activate(this, &staticMetaObject, 0, nullptr);
}
struct qt_meta_stringdata_AbstractControlSP_t {
    QByteArrayData data[10];
    char stringdata0[154];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_AbstractControlSP_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_AbstractControlSP_t qt_meta_stringdata_AbstractControlSP = {
    {
QT_MOC_LITERAL(0, 0, 17), // "AbstractControlSP"
QT_MOC_LITERAL(1, 18, 15), // "showDescription"
QT_MOC_LITERAL(2, 34, 0), // ""
QT_MOC_LITERAL(3, 35, 10), // "setVisible"
QT_MOC_LITERAL(4, 46, 7), // "visible"
QT_MOC_LITERAL(5, 54, 23), // "RegisterAdvancedControl"
QT_MOC_LITERAL(6, 78, 18), // "AbstractControlSP*"
QT_MOC_LITERAL(7, 97, 4), // "ctrl"
QT_MOC_LITERAL(8, 102, 25), // "UnregisterAdvancedControl"
QT_MOC_LITERAL(9, 128, 25) // "UpdateAllAdvancedControls"

    },
    "AbstractControlSP\0showDescription\0\0"
    "setVisible\0visible\0RegisterAdvancedControl\0"
    "AbstractControlSP*\0ctrl\0"
    "UnregisterAdvancedControl\0"
    "UpdateAllAdvancedControls"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_AbstractControlSP[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       5,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    0,   39,    2, 0x0a /* Public */,
       3,    1,   40,    2, 0x0a /* Public */,
       5,    1,   43,    2, 0x0a /* Public */,
       8,    1,   46,    2, 0x0a /* Public */,
       9,    0,   49,    2, 0x0a /* Public */,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void, QMetaType::Bool,    4,
    QMetaType::Void, 0x80000000 | 6,    7,
    QMetaType::Void, 0x80000000 | 6,    7,
    QMetaType::Void,

       0        // eod
};

void AbstractControlSP::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<AbstractControlSP *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->showDescription(); break;
        case 1: _t->setVisible((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 2: _t->RegisterAdvancedControl((*reinterpret_cast< AbstractControlSP*(*)>(_a[1]))); break;
        case 3: _t->UnregisterAdvancedControl((*reinterpret_cast< AbstractControlSP*(*)>(_a[1]))); break;
        case 4: _t->UpdateAllAdvancedControls(); break;
        default: ;
        }
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        switch (_id) {
        default: *reinterpret_cast<int*>(_a[0]) = -1; break;
        case 2:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< AbstractControlSP* >(); break;
            }
            break;
        case 3:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< AbstractControlSP* >(); break;
            }
            break;
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject AbstractControlSP::staticMetaObject = { {
    &AbstractControl::staticMetaObject,
    qt_meta_stringdata_AbstractControlSP.data,
    qt_meta_data_AbstractControlSP,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *AbstractControlSP::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *AbstractControlSP::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_AbstractControlSP.stringdata0))
        return static_cast<void*>(this);
    return AbstractControl::qt_metacast(_clname);
}

int AbstractControlSP::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = AbstractControl::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 5)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 5;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 5)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 5;
    }
    return _id;
}
struct qt_meta_stringdata_AbstractControlSP_SELECTOR_t {
    QByteArrayData data[1];
    char stringdata0[27];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_AbstractControlSP_SELECTOR_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_AbstractControlSP_SELECTOR_t qt_meta_stringdata_AbstractControlSP_SELECTOR = {
    {
QT_MOC_LITERAL(0, 0, 26) // "AbstractControlSP_SELECTOR"

    },
    "AbstractControlSP_SELECTOR"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_AbstractControlSP_SELECTOR[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       0,    0, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

       0        // eod
};

void AbstractControlSP_SELECTOR::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    Q_UNUSED(_o);
    Q_UNUSED(_id);
    Q_UNUSED(_c);
    Q_UNUSED(_a);
}

QT_INIT_METAOBJECT const QMetaObject AbstractControlSP_SELECTOR::staticMetaObject = { {
    &AbstractControlSP::staticMetaObject,
    qt_meta_stringdata_AbstractControlSP_SELECTOR.data,
    qt_meta_data_AbstractControlSP_SELECTOR,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *AbstractControlSP_SELECTOR::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *AbstractControlSP_SELECTOR::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_AbstractControlSP_SELECTOR.stringdata0))
        return static_cast<void*>(this);
    return AbstractControlSP::qt_metacast(_clname);
}

int AbstractControlSP_SELECTOR::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = AbstractControlSP::qt_metacall(_c, _id, _a);
    return _id;
}
struct qt_meta_stringdata_LabelControlSP_t {
    QByteArrayData data[1];
    char stringdata0[15];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_LabelControlSP_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_LabelControlSP_t qt_meta_stringdata_LabelControlSP = {
    {
QT_MOC_LITERAL(0, 0, 14) // "LabelControlSP"

    },
    "LabelControlSP"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_LabelControlSP[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       0,    0, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

       0        // eod
};

void LabelControlSP::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    Q_UNUSED(_o);
    Q_UNUSED(_id);
    Q_UNUSED(_c);
    Q_UNUSED(_a);
}

QT_INIT_METAOBJECT const QMetaObject LabelControlSP::staticMetaObject = { {
    &AbstractControlSP::staticMetaObject,
    qt_meta_stringdata_LabelControlSP.data,
    qt_meta_data_LabelControlSP,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *LabelControlSP::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *LabelControlSP::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_LabelControlSP.stringdata0))
        return static_cast<void*>(this);
    return AbstractControlSP::qt_metacast(_clname);
}

int LabelControlSP::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = AbstractControlSP::qt_metacall(_c, _id, _a);
    return _id;
}
struct qt_meta_stringdata_ButtonControlSP_t {
    QByteArrayData data[5];
    char stringdata0[44];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_ButtonControlSP_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_ButtonControlSP_t qt_meta_stringdata_ButtonControlSP = {
    {
QT_MOC_LITERAL(0, 0, 15), // "ButtonControlSP"
QT_MOC_LITERAL(1, 16, 7), // "clicked"
QT_MOC_LITERAL(2, 24, 0), // ""
QT_MOC_LITERAL(3, 25, 10), // "setEnabled"
QT_MOC_LITERAL(4, 36, 7) // "enabled"

    },
    "ButtonControlSP\0clicked\0\0setEnabled\0"
    "enabled"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_ButtonControlSP[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       2,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    0,   24,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
       3,    1,   25,    2, 0x0a /* Public */,

 // signals: parameters
    QMetaType::Void,

 // slots: parameters
    QMetaType::Void, QMetaType::Bool,    4,

       0        // eod
};

void ButtonControlSP::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<ButtonControlSP *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->clicked(); break;
        case 1: _t->setEnabled((*reinterpret_cast< bool(*)>(_a[1]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (ButtonControlSP::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&ButtonControlSP::clicked)) {
                *result = 0;
                return;
            }
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject ButtonControlSP::staticMetaObject = { {
    &AbstractControlSP::staticMetaObject,
    qt_meta_stringdata_ButtonControlSP.data,
    qt_meta_data_ButtonControlSP,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *ButtonControlSP::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *ButtonControlSP::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_ButtonControlSP.stringdata0))
        return static_cast<void*>(this);
    return AbstractControlSP::qt_metacast(_clname);
}

int ButtonControlSP::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = AbstractControlSP::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 2)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 2;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 2)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 2;
    }
    return _id;
}

// SIGNAL 0
void ButtonControlSP::clicked()
{
    QMetaObject::activate(this, &staticMetaObject, 0, nullptr);
}
struct qt_meta_stringdata_ToggleControlSP_t {
    QByteArrayData data[4];
    char stringdata0[37];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_ToggleControlSP_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_ToggleControlSP_t qt_meta_stringdata_ToggleControlSP = {
    {
QT_MOC_LITERAL(0, 0, 15), // "ToggleControlSP"
QT_MOC_LITERAL(1, 16, 13), // "toggleFlipped"
QT_MOC_LITERAL(2, 30, 0), // ""
QT_MOC_LITERAL(3, 31, 5) // "state"

    },
    "ToggleControlSP\0toggleFlipped\0\0state"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_ToggleControlSP[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       1,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    1,   19,    2, 0x06 /* Public */,

 // signals: parameters
    QMetaType::Void, QMetaType::Bool,    3,

       0        // eod
};

void ToggleControlSP::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<ToggleControlSP *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->toggleFlipped((*reinterpret_cast< bool(*)>(_a[1]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (ToggleControlSP::*)(bool );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&ToggleControlSP::toggleFlipped)) {
                *result = 0;
                return;
            }
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject ToggleControlSP::staticMetaObject = { {
    &AbstractControlSP::staticMetaObject,
    qt_meta_stringdata_ToggleControlSP.data,
    qt_meta_data_ToggleControlSP,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *ToggleControlSP::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *ToggleControlSP::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_ToggleControlSP.stringdata0))
        return static_cast<void*>(this);
    return AbstractControlSP::qt_metacast(_clname);
}

int ToggleControlSP::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = AbstractControlSP::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 1)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 1;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 1)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 1;
    }
    return _id;
}

// SIGNAL 0
void ToggleControlSP::toggleFlipped(bool _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}
struct qt_meta_stringdata_ParamControlSP_t {
    QByteArrayData data[1];
    char stringdata0[15];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_ParamControlSP_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_ParamControlSP_t qt_meta_stringdata_ParamControlSP = {
    {
QT_MOC_LITERAL(0, 0, 14) // "ParamControlSP"

    },
    "ParamControlSP"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_ParamControlSP[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       0,    0, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

       0        // eod
};

void ParamControlSP::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    Q_UNUSED(_o);
    Q_UNUSED(_id);
    Q_UNUSED(_c);
    Q_UNUSED(_a);
}

QT_INIT_METAOBJECT const QMetaObject ParamControlSP::staticMetaObject = { {
    &ToggleControlSP::staticMetaObject,
    qt_meta_stringdata_ParamControlSP.data,
    qt_meta_data_ParamControlSP,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *ParamControlSP::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *ParamControlSP::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_ParamControlSP.stringdata0))
        return static_cast<void*>(this);
    return ToggleControlSP::qt_metacast(_clname);
}

int ParamControlSP::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = ToggleControlSP::qt_metacall(_c, _id, _a);
    return _id;
}
struct qt_meta_stringdata_MultiButtonControlSP_t {
    QByteArrayData data[4];
    char stringdata0[39];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_MultiButtonControlSP_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_MultiButtonControlSP_t qt_meta_stringdata_MultiButtonControlSP = {
    {
QT_MOC_LITERAL(0, 0, 20), // "MultiButtonControlSP"
QT_MOC_LITERAL(1, 21, 13), // "buttonClicked"
QT_MOC_LITERAL(2, 35, 0), // ""
QT_MOC_LITERAL(3, 36, 2) // "id"

    },
    "MultiButtonControlSP\0buttonClicked\0\0"
    "id"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_MultiButtonControlSP[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       1,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    1,   19,    2, 0x06 /* Public */,

 // signals: parameters
    QMetaType::Void, QMetaType::Int,    3,

       0        // eod
};

void MultiButtonControlSP::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<MultiButtonControlSP *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->buttonClicked((*reinterpret_cast< int(*)>(_a[1]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (MultiButtonControlSP::*)(int );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&MultiButtonControlSP::buttonClicked)) {
                *result = 0;
                return;
            }
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject MultiButtonControlSP::staticMetaObject = { {
    &AbstractControlSP_SELECTOR::staticMetaObject,
    qt_meta_stringdata_MultiButtonControlSP.data,
    qt_meta_data_MultiButtonControlSP,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *MultiButtonControlSP::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *MultiButtonControlSP::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_MultiButtonControlSP.stringdata0))
        return static_cast<void*>(this);
    return AbstractControlSP_SELECTOR::qt_metacast(_clname);
}

int MultiButtonControlSP::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = AbstractControlSP_SELECTOR::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 1)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 1;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 1)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 1;
    }
    return _id;
}

// SIGNAL 0
void MultiButtonControlSP::buttonClicked(int _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}
struct qt_meta_stringdata_ButtonParamControlSP_t {
    QByteArrayData data[1];
    char stringdata0[21];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_ButtonParamControlSP_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_ButtonParamControlSP_t qt_meta_stringdata_ButtonParamControlSP = {
    {
QT_MOC_LITERAL(0, 0, 20) // "ButtonParamControlSP"

    },
    "ButtonParamControlSP"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_ButtonParamControlSP[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       0,    0, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

       0        // eod
};

void ButtonParamControlSP::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    Q_UNUSED(_o);
    Q_UNUSED(_id);
    Q_UNUSED(_c);
    Q_UNUSED(_a);
}

QT_INIT_METAOBJECT const QMetaObject ButtonParamControlSP::staticMetaObject = { {
    &MultiButtonControlSP::staticMetaObject,
    qt_meta_stringdata_ButtonParamControlSP.data,
    qt_meta_data_ButtonParamControlSP,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *ButtonParamControlSP::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *ButtonParamControlSP::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_ButtonParamControlSP.stringdata0))
        return static_cast<void*>(this);
    return MultiButtonControlSP::qt_metacast(_clname);
}

int ButtonParamControlSP::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = MultiButtonControlSP::qt_metacall(_c, _id, _a);
    return _id;
}
struct qt_meta_stringdata_ListWidgetSP_t {
    QByteArrayData data[1];
    char stringdata0[13];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_ListWidgetSP_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_ListWidgetSP_t qt_meta_stringdata_ListWidgetSP = {
    {
QT_MOC_LITERAL(0, 0, 12) // "ListWidgetSP"

    },
    "ListWidgetSP"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_ListWidgetSP[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       0,    0, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

       0        // eod
};

void ListWidgetSP::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    Q_UNUSED(_o);
    Q_UNUSED(_id);
    Q_UNUSED(_c);
    Q_UNUSED(_a);
}

QT_INIT_METAOBJECT const QMetaObject ListWidgetSP::staticMetaObject = { {
    &QWidget::staticMetaObject,
    qt_meta_stringdata_ListWidgetSP.data,
    qt_meta_data_ListWidgetSP,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *ListWidgetSP::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *ListWidgetSP::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_ListWidgetSP.stringdata0))
        return static_cast<void*>(this);
    return QWidget::qt_metacast(_clname);
}

int ListWidgetSP::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    return _id;
}
struct qt_meta_stringdata_LayoutWidgetSP_t {
    QByteArrayData data[1];
    char stringdata0[15];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_LayoutWidgetSP_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_LayoutWidgetSP_t qt_meta_stringdata_LayoutWidgetSP = {
    {
QT_MOC_LITERAL(0, 0, 14) // "LayoutWidgetSP"

    },
    "LayoutWidgetSP"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_LayoutWidgetSP[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       0,    0, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

       0        // eod
};

void LayoutWidgetSP::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    Q_UNUSED(_o);
    Q_UNUSED(_id);
    Q_UNUSED(_c);
    Q_UNUSED(_a);
}

QT_INIT_METAOBJECT const QMetaObject LayoutWidgetSP::staticMetaObject = { {
    &QWidget::staticMetaObject,
    qt_meta_stringdata_LayoutWidgetSP.data,
    qt_meta_data_LayoutWidgetSP,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *LayoutWidgetSP::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *LayoutWidgetSP::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_LayoutWidgetSP.stringdata0))
        return static_cast<void*>(this);
    return QWidget::qt_metacast(_clname);
}

int LayoutWidgetSP::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    return _id;
}
struct qt_meta_stringdata_OptionControlSP_t {
    QByteArrayData data[4];
    char stringdata0[49];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_OptionControlSP_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_OptionControlSP_t qt_meta_stringdata_OptionControlSP = {
    {
QT_MOC_LITERAL(0, 0, 15), // "OptionControlSP"
QT_MOC_LITERAL(1, 16, 12), // "updateLabels"
QT_MOC_LITERAL(2, 29, 0), // ""
QT_MOC_LITERAL(3, 30, 18) // "updateOtherToggles"

    },
    "OptionControlSP\0updateLabels\0\0"
    "updateOtherToggles"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_OptionControlSP[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       2,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       2,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    0,   24,    2, 0x06 /* Public */,
       3,    0,   25,    2, 0x06 /* Public */,

 // signals: parameters
    QMetaType::Void,
    QMetaType::Void,

       0        // eod
};

void OptionControlSP::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<OptionControlSP *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->updateLabels(); break;
        case 1: _t->updateOtherToggles(); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (OptionControlSP::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&OptionControlSP::updateLabels)) {
                *result = 0;
                return;
            }
        }
        {
            using _t = void (OptionControlSP::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&OptionControlSP::updateOtherToggles)) {
                *result = 1;
                return;
            }
        }
    }
    Q_UNUSED(_a);
}

QT_INIT_METAOBJECT const QMetaObject OptionControlSP::staticMetaObject = { {
    &AbstractControlSP_SELECTOR::staticMetaObject,
    qt_meta_stringdata_OptionControlSP.data,
    qt_meta_data_OptionControlSP,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *OptionControlSP::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *OptionControlSP::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_OptionControlSP.stringdata0))
        return static_cast<void*>(this);
    return AbstractControlSP_SELECTOR::qt_metacast(_clname);
}

int OptionControlSP::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = AbstractControlSP_SELECTOR::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 2)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 2;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 2)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 2;
    }
    return _id;
}

// SIGNAL 0
void OptionControlSP::updateLabels()
{
    QMetaObject::activate(this, &staticMetaObject, 0, nullptr);
}

// SIGNAL 1
void OptionControlSP::updateOtherToggles()
{
    QMetaObject::activate(this, &staticMetaObject, 1, nullptr);
}
struct qt_meta_stringdata_PushButtonSP_t {
    QByteArrayData data[1];
    char stringdata0[13];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_PushButtonSP_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_PushButtonSP_t qt_meta_stringdata_PushButtonSP = {
    {
QT_MOC_LITERAL(0, 0, 12) // "PushButtonSP"

    },
    "PushButtonSP"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_PushButtonSP[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       0,    0, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

       0        // eod
};

void PushButtonSP::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    Q_UNUSED(_o);
    Q_UNUSED(_id);
    Q_UNUSED(_c);
    Q_UNUSED(_a);
}

QT_INIT_METAOBJECT const QMetaObject PushButtonSP::staticMetaObject = { {
    &QPushButton::staticMetaObject,
    qt_meta_stringdata_PushButtonSP.data,
    qt_meta_data_PushButtonSP,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *PushButtonSP::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *PushButtonSP::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_PushButtonSP.stringdata0))
        return static_cast<void*>(this);
    return QPushButton::qt_metacast(_clname);
}

int PushButtonSP::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QPushButton::qt_metacall(_c, _id, _a);
    return _id;
}
struct qt_meta_stringdata_PanelBackButton_t {
    QByteArrayData data[1];
    char stringdata0[16];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_PanelBackButton_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_PanelBackButton_t qt_meta_stringdata_PanelBackButton = {
    {
QT_MOC_LITERAL(0, 0, 15) // "PanelBackButton"

    },
    "PanelBackButton"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_PanelBackButton[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       0,    0, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

       0        // eod
};

void PanelBackButton::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    Q_UNUSED(_o);
    Q_UNUSED(_id);
    Q_UNUSED(_c);
    Q_UNUSED(_a);
}

QT_INIT_METAOBJECT const QMetaObject PanelBackButton::staticMetaObject = { {
    &QPushButton::staticMetaObject,
    qt_meta_stringdata_PanelBackButton.data,
    qt_meta_data_PanelBackButton,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *PanelBackButton::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *PanelBackButton::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_PanelBackButton.stringdata0))
        return static_cast<void*>(this);
    return QPushButton::qt_metacast(_clname);
}

int PanelBackButton::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QPushButton::qt_metacall(_c, _id, _a);
    return _id;
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
