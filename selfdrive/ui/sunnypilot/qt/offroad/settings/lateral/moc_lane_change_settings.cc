/****************************************************************************
** Meta object code from reading C++ file 'lane_change_settings.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.12.8)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "lane_change_settings.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'lane_change_settings.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.12.8. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_AutoLaneChangeTimer_t {
    QByteArrayData data[3];
    char stringdata0[35];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_AutoLaneChangeTimer_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_AutoLaneChangeTimer_t qt_meta_stringdata_AutoLaneChangeTimer = {
    {
QT_MOC_LITERAL(0, 0, 19), // "AutoLaneChangeTimer"
QT_MOC_LITERAL(1, 20, 13), // "toggleUpdated"
QT_MOC_LITERAL(2, 34, 0) // ""

    },
    "AutoLaneChangeTimer\0toggleUpdated\0"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_AutoLaneChangeTimer[] = {

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

void AutoLaneChangeTimer::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<AutoLaneChangeTimer *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->toggleUpdated(); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (AutoLaneChangeTimer::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&AutoLaneChangeTimer::toggleUpdated)) {
                *result = 0;
                return;
            }
        }
    }
    Q_UNUSED(_a);
}

QT_INIT_METAOBJECT const QMetaObject AutoLaneChangeTimer::staticMetaObject = { {
    &OptionControlSP::staticMetaObject,
    qt_meta_stringdata_AutoLaneChangeTimer.data,
    qt_meta_data_AutoLaneChangeTimer,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *AutoLaneChangeTimer::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *AutoLaneChangeTimer::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_AutoLaneChangeTimer.stringdata0))
        return static_cast<void*>(this);
    return OptionControlSP::qt_metacast(_clname);
}

int AutoLaneChangeTimer::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = OptionControlSP::qt_metacall(_c, _id, _a);
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
void AutoLaneChangeTimer::toggleUpdated()
{
    QMetaObject::activate(this, &staticMetaObject, 0, nullptr);
}
struct qt_meta_stringdata_LaneChangeSettings_t {
    QByteArrayData data[4];
    char stringdata0[44];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_LaneChangeSettings_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_LaneChangeSettings_t qt_meta_stringdata_LaneChangeSettings = {
    {
QT_MOC_LITERAL(0, 0, 18), // "LaneChangeSettings"
QT_MOC_LITERAL(1, 19, 9), // "backPress"
QT_MOC_LITERAL(2, 29, 0), // ""
QT_MOC_LITERAL(3, 30, 13) // "updateToggles"

    },
    "LaneChangeSettings\0backPress\0\0"
    "updateToggles"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_LaneChangeSettings[] = {

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
       3,    0,   25,    2, 0x0a /* Public */,

 // signals: parameters
    QMetaType::Void,

 // slots: parameters
    QMetaType::Void,

       0        // eod
};

void LaneChangeSettings::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<LaneChangeSettings *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->backPress(); break;
        case 1: _t->updateToggles(); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (LaneChangeSettings::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&LaneChangeSettings::backPress)) {
                *result = 0;
                return;
            }
        }
    }
    Q_UNUSED(_a);
}

QT_INIT_METAOBJECT const QMetaObject LaneChangeSettings::staticMetaObject = { {
    &QWidget::staticMetaObject,
    qt_meta_stringdata_LaneChangeSettings.data,
    qt_meta_data_LaneChangeSettings,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *LaneChangeSettings::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *LaneChangeSettings::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_LaneChangeSettings.stringdata0))
        return static_cast<void*>(this);
    return QWidget::qt_metacast(_clname);
}

int LaneChangeSettings::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
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
void LaneChangeSettings::backPress()
{
    QMetaObject::activate(this, &staticMetaObject, 0, nullptr);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
